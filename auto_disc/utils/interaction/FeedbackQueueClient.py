#!/usr/bin/env python3
"""Module containing clients for interacting with a file-based message queue
used to broker communication with the user who can provide feedback to the
exploration loop via ansewring questions asynchronously."""
import asyncio
import os
import pickle
import re
import stat
import tempfile
import traceback
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Optional, Tuple
from uuid import uuid4 as generate_uuid

from pexpect import pxssh
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


@dataclass
class Feedback:
    """Class whose objects are instances of human feedback to the exploration
    loop."""

    content: Dict
    id: Optional[int] = field(default_factory=lambda: generate_uuid().int)

    def __hash__(self):
        # NOTE: that defining the hash function in this way asserts that
        # no two Feedback objects are ever equal
        return self.id


class _FeedbackQueueClient:
    """Abstract class whose instances are connection clients to a file-based message
    queue allowing two-way communication about feedbacks.

    NOTE: this class does not inherit from `Leaf` because the persistence of its
    state is already guaranteed by the existence of the file-based message queue.
    """

    def __init__(self, persist_path: str = "/tmp/messageq") -> None:
        # setup directories for the queue
        self.question_path = os.path.join(persist_path, "questions")
        self.response_path = os.path.join(persist_path, "responses")
        os.makedirs(self.question_path, exist_ok=True)
        os.makedirs(self.response_path, exist_ok=True)

        # create in-memory queues
        self.questions = Queue()
        self.responses = Queue()

    def get_question(self, timeout: int = 5, block: bool = True) -> Feedback:
        """Get a question from the in-memory cache.

        NOTE: synchronization with the disk is not handled.

        Args
            timeout: Timeout length in seconds.
            block: Whether or not to block the thread when getting.

        Returns
            Feedback object retrieved from the queue.
        """

        return self.questions.get(block=block, timeout=timeout)

    def put_question(self, question: Feedback) -> None:
        """Put a question to disk and synchronize with in-memory cache."""
        raise NotImplementedError

    def get_response(self, timeout: int = 5, block: bool = True):
        """Get a response from the in-memory cache.
        NOTE: synchronization with the disk is not handled.

        Args
            timeout: Timeout length in seconds.
            block: Whether or not to block the thread when getting.

        Returns
            Feedback object retrieved from the queue.
        """
        return self.responses.get(block=block, timeout=timeout)

    def put_response(self, response: Feedback):
        """Put a response to disk and synchronize with in-memory cache."""
        raise NotImplementedError

    def listen_for_questions(self):
        """Asynchronously watch the questions queue and add new `Feedback`s to
        in-memory queue.

        NOTE: that it will only listen for new `Feedback`s
        added after being called and does not examine what is persisted on disk.

        Return
            A handle which can be used to terminate the watch by calling its
                shutdown() method
        Raises
            FileNotFoundError: Could not find directory dir.
        """
        return self.watch_directory(dir=self.question_path, queue=self.questions)

    def listen_for_responses(self):
        """Asynchronously watch the questions queue and add new `Feedback`s to
        in-memory queue.

        NOTE: that it will only listen for new `Feedback`s
        added after being called and does not examine what is persisted on disk.

        Return
            A handle which can be used to terminate the watch by calling its
                shutdown() method
        Raises
            FileNotFoundError: Could not find directory dir.
        """
        return self.watch_directory(dir=self.response_path, queue=self.responses)

    @staticmethod
    def watch_directory(dir: str, queue: Queue):
        """Asynchronously watch a directory and add new file paths to queue.

        For performance reasons, the opening of files are to be handled by the
        caller, however the `Feedback` queues are such that each feedback is
        read-only on disk.

        Args
            dir: Path to the directory to watch.
            queue: In-memory queue to which file paths will be added.

        Return
            A handle which can be used to terminate the watch by calling its
                shutdown() method
        Raises
            FileNotFoundError: Could not find directory dir.
        """
        raise NotImplementedError


class LocalQueueClient(_FeedbackQueueClient):
    """Reified subclass of _FeedbackQueueClient whose instances are connection
    clients connected to a queue stored locally.
    """

    def put_question(self, question: Feedback):
        # persist feedback to disk
        file_to_write = os.path.join(self.question_path, str(question.id))
        with open(file_to_write, "wb") as f:
            pickle.dump(question, f)

        # set read-only
        os.chmod(file_to_write, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # put into the in-memory queue
        self.questions.put(question)

        return

    def put_response(self, response: Feedback):
        # persist feedback to disk
        file_to_write = os.path.join(self.response_path, str(response.id))
        with open(file_to_write, "wb") as f:
            pickle.dump(response, f)

        # set read-only
        os.chmod(file_to_write, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # put into the in-memory queue
        self.responses.put(response)

        return

    @staticmethod
    def watch_directory(dir: str, queue: Queue):
        if not os.path.isdir(dir):
            raise FileNotFoundError(
                f"Could not open directory with name {dir}, it may not exist."
            )

        class QueueHandler(FileSystemEventHandler):
            def __init__(self, q: Queue):
                super().__init__()
                self.queue = q

            def on_created(self, event):
                # only act on file creations
                if event.is_directory == False:
                    with open(event.src_path, "rb") as f:
                        self.queue.put(pickle.load(f))

        observer = Observer()
        queue_handler = QueueHandler(queue)
        observer.schedule(queue_handler, dir, recursive=False)
        observer.start()

        return observer


class RemoteQueueClient(_FeedbackQueueClient):
    """Reified subclass of _FeedbackQueueClient whose instances are connection
    clients connected to a queue stored on a remote host accessible by SSH.
    """

    def __init__(
        self,
        persist_path: str = "me@example.com/tmp/messageq",
        ssh_config: str = "./config",
    ) -> None:
        # parse ssh info
        self.user, self.host, self.persist_path = _parse_ssh_url(persist_path)

        # TODO: make this lazy evaluate when needed on method calls instead of
        # at init
        # create SSH connection
        self.shell = pxssh.pxssh()
        self.shell.login(
            self.host,
            ssh_config=ssh_config,
        )

        # setup directories for the queue
        self.question_path = os.path.join(self.persist_path, "questions")
        self.response_path = os.path.join(self.persist_path, "responses")
        self._initialize_directories(self.question_path, self.response_path)

        # create in-memory queues
        self.questions = Queue()
        self.responses = Queue()

    def _initialize_directories(self, question_path: str, response_path: str):
        self.shell.sendline(f"mkdir -p {self.question_path} {self.response_path}")
        self.shell.prompt()

        # create a dummy file so that the watcher will never poll an empty
        # directory
        self.shell.sendline(
            f"touch {self.question_path.rstrip('/')}/queue {self.response_path.rstrip('/')}/queue"
        )

    def _push_file(self, local_file: str, remote_file: str):
        if self.user is None:
            connection_str = f"{self.host}"
        else:
            connection_str = f"{self.user}@{self.host}"

        os.system(
            "scp -r {} {}:{}".format(
                local_file,
                connection_str,
                remote_file,
            )
        )

    def _pull_file(self, remote_file: str, local_file: str):
        if self.user is None:
            connection_str = f"{self.host}"
        else:
            connection_str = f"{self.user}@{self.host}"
        os.system(
            "scp -r {}:{} {}".format(
                connection_str,
                remote_file,
                local_file,
            )
        )

    def put_question(self, question: Feedback):
        # persist feedback to disk on remote
        file_to_write = os.path.join(self.question_path, str(question.id))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "pkl")
            with open(tmpfile, "wb") as f:
                pickle.dump(question, f)
            self._push_file(tmpfile, file_to_write)

        # set read-only
        self.shell.sendline(f"chmod 444 {file_to_write}")

        # put into the in-memory queue
        self.questions.put(question)

        return

    def put_response(self, response: Feedback):
        # persist feedback to disk on remote
        file_to_write = os.path.join(self.response_path, str(response.id))
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(response, f)
            self._push_file(f.name, file_to_write)
        # set read-only
        self.shell.sendline(f"chmod 444 {file_to_write}")

        # put into the in-memory queue
        self.responses.put(response)

        return

    def watch_directory(self, dir: str, queue: Queue, poll_interval: int = 1):
        def poll_newest_filename(dir):
            self.shell.sendline(f"ls -c {dir} | head -n 1")
            self.shell.prompt()
            if self.shell.before is None:
                raise Exception("Could not parse result of SSH command.")
            else:
                return os.path.join(dir, self.shell.before.decode().split("\r\n")[1])

        async def retry_coroutine(coro, *args, **kwargs):
            while True:
                try:
                    await coro(*args, **kwargs)
                except asyncio.CancelledError:
                    # pass through legitimate cancellations
                    raise
                except Exception:
                    print("Caught exception in task")
                    traceback.print_exc()

        async def poll():
            buffered_file = ""
            while True:
                await asyncio.sleep(poll_interval)
                # get the name of the newest created file
                polled_file = poll_newest_filename(dir)
                print(f"Polled file {polled_file}")
                # check if
                # 1. file is updated
                # 2. file is not the root dir (pxssh parsing bug), trailing
                #   slash stripped due to conflicting conventions about dir path
                #   names
                # 3. file is not the default placeholder file
                if (
                    (polled_file != buffered_file)
                    and (polled_file.rstrip("/") != dir.rstrip("/"))
                    and (os.path.basename(polled_file) != "queue")
                ):
                    buffered_file = polled_file
                    print(f"Buffered file is now {buffered_file}")
                    with tempfile.TemporaryDirectory() as tmpdir:
                        self._pull_file(polled_file, tmpdir)
                        tmpfile = os.path.join(tmpdir, os.path.basename(polled_file))
                        with open(tmpfile, "rb") as f:
                            feedback = pickle.load(f)
                            queue.put(feedback)

        task = asyncio.create_task(retry_coroutine(poll))

        # alias the .cancel() method to obey the interface
        task.shutdown = task.cancel

        # TODO: make it handle keyboard interrupts
        return task


def make_FeedbackQueueClient(persist_url: str, *args, **kwargs) -> _FeedbackQueueClient:
    """Make connection client object.

    Args
        persist_url:
            URL for the persistence path, i.e., a path-name with optional
            protocol prefix such as `ssh:///tmp/messageq` or
            `file:///tmp/messageq`

    Returns
        Instantiated object of the feedback queue connection client, dispatched
        based on the protocol.

    Raises
        ValueError: Given `persist_url` does not match implemented protocols.
    """
    path, protocol_name = _get_protocol(persist_url)
    if protocol_name not in ["ssh", "sftp", "file"]:
        raise ValueError("Given `persist_url` does not match implemented protocols.")

    if protocol_name == "file":
        return LocalQueueClient(path)
    elif protocol_name in ["ssh", "sftp"]:
        return RemoteQueueClient(path, *args, **kwargs)


def _get_protocol(url: str) -> Tuple[str, str]:
    if url.find(":") == -1:
        protocol_name = "file"
        output_path = url
    else:
        protocol_name = url[: url.find(":")]
        output_path = url[url.find(":") + 3 :]

    return output_path, protocol_name


def _parse_ssh_url(stripped_url: str) -> Tuple[str, str, str]:
    res = re.match(r"^(?:(?P<user>\w*)@)?(?P<host>.*?)(?P<path>\/.*)", stripped_url)
    if res is None:
        raise Exception("Error parsing SSH url.")
    user, host, path = res.group("user", "host", "path")
    return user, host, path
