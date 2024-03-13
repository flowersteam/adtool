#!/usr/bin/env python3
import concurrent.futures
import logging
import os
import queue
import shutil
from time import sleep
from uuid import uuid4

import pytest
from adtool.utils.interaction.FeedbackQueueClient import (
    Feedback,
    LocalQueueClient,
    RemoteQueueClient,
    _get_protocol,
    make_FeedbackQueueClient,
)

global PERSIST_PATH
PERSIST_PATH = os.path.join("/tmp", uuid4().hex)


def setup_function(function):
    os.mkdir(PERSIST_PATH)


def teardown_function(function):
    shutil.rmtree(PERSIST_PATH)


def test__get_protocol():
    test_paths = [
        "/tmp/messageq",
        "file:///tmp/messageq",
        "ssh:///tmp/messageq",
        "sftp:///tmp/messageq",
        "ftp:///tmp/messageq",
        "ssh://me@example.com/tmp/messageq",
    ]

    answers = [
        ("/tmp/messageq", "file"),
        ("/tmp/messageq", "file"),
        ("/tmp/messageq", "ssh"),
        ("/tmp/messageq", "sftp"),
        ("/tmp/messageq", "ftp"),
        ("me@example.com/tmp/messageq", "ssh"),
    ]

    for a, p in zip(answers, test_paths):
        assert a == _get_protocol(p)


def test_make_FeedbackQueueClient(mocker):
    # exception branch
    test_path = "ftp:///tmp/messageq"
    with pytest.raises(ValueError):
        make_FeedbackQueueClient(test_path)

    # dispatches
    test_path = "ssh:///tmp/messageq"
    mocker.patch("pexpect.pxssh.pxssh")
    assert type(make_FeedbackQueueClient(test_path)) == RemoteQueueClient

    test_path = "file:///tmp/messageq"
    assert type(make_FeedbackQueueClient(test_path)) == LocalQueueClient


def test_LocalQueueClient___init__():
    client = make_FeedbackQueueClient(PERSIST_PATH)
    assert os.path.exists(os.path.join(PERSIST_PATH, "questions"))
    assert os.path.exists(os.path.join(PERSIST_PATH, "responses"))


def test_LocalQueueClient_queue_operations():
    f = [Feedback({"a": 1}), Feedback({"a": 2})]
    client = make_FeedbackQueueClient(PERSIST_PATH)
    client.put_question(f[0])
    client.put_question(f[1])
    client.put_response(f[1])
    client.put_response(f[0])
    assert client.get_question() == f[0]
    assert client.get_question() == f[1]
    assert client.get_response() == f[1]
    assert client.get_response() == f[0]


def test_LocalQueueClient_queue_operations_empty():
    client = make_FeedbackQueueClient(PERSIST_PATH)
    with pytest.raises(queue.Empty):
        client.get_question(timeout=1)


def test_LocalQueueClient_listener():
    client = make_FeedbackQueueClient(PERSIST_PATH)
    handler = client.listen_for_questions()

    put_results = []

    # need to spin a subprocess due to GIL
    def delayed_put():
        import logging

        client = make_FeedbackQueueClient(PERSIST_PATH)
        for i in range(5):
            sleep(2)
            logging.info(f"Pushing {i}")
            fb = Feedback(content={"index": i})
            client.put_question(fb)
            put_results.append(fb)
            logging.info(f"Pushed {fb}")

    # single background provider
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(delayed_put)

        # compare results which are polled asynchronously
        poll_results = []
        while len(poll_results) < 5:
            logging.info("Polling results.")
            poll_results.append(client.get_question(timeout=10))
            logging.info(f"Have polled {len(poll_results)} results.")

        assert poll_results == put_results

    # multiple background providers
    put_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(delayed_put)
        sleep(3)
        executor.submit(delayed_put)
        sleep(4)
        executor.submit(delayed_put)

        # compare results which are polled asynchronously
        poll_results = []
        while len(poll_results) < 15:
            poll_results.append(client.get_question(timeout=10))
            logging.info(f"Have polled {len(poll_results)} results.")

        assert set(poll_results) == set(put_results)
