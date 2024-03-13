from __future__ import annotations

import logging
from random import seed
from typing import List


class AutoDiscLogger(logging.Logger):
    """
    A logger to manage experiments logs and print them in the console or save them to the database or to disk according to the user's needs.
    """

    def __init__(
        self, experiment_id: int, seed: int, handlers: List[AutoDiscLogger]
    ) -> None:
        """
        Init the logger for an experiment.

        Args:
            experiment_id: current experiment id
            seed: current seed number
            handlers: List of all handlers needed by the user to manage logs (e.g. save in database on disk)
        """
        self.__experiment_id = experiment_id
        self._seed = seed
        self._shared_logger = logging.getLogger("ad_tool_logger")
        self.__index = 0
        # create handler
        # add handler
        if not any(
            handler.experiment_id == self.__experiment_id
            for handler in self._shared_logger.handlers
        ):
            stream_h = logging.StreamHandler()
            stream_h.setLevel(logging.NOTSET)
            logging.getLogger().setLevel(logging.NOTSET)
            formatter = logging.Formatter(
                "%(name)s - %(levelname)s - SEED %(seed)s - LOG_ID %(id)s - %(message)s"
            )
            stream_h.setFormatter(formatter)
            stream_h.experiment_id = experiment_id
            self._shared_logger.addHandler(stream_h)
            for handler in handlers:
                self._shared_logger.addHandler(handler)
            self._shared_logger.addFilter(ContextFilter())

    def debug(self, *args) -> None:
        """
        Call the logs method at the debug level and increment the log index to make an unique id for each log
        """
        self.__index += 1
        self._shared_logger.debug(
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )

    def info(self, *args) -> None:
        """
        Call the logs method at the info level and increment the log index to make an unique id for each log
        """
        self.__index += 1
        self._shared_logger.info(
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )

    def warning(self, *args) -> None:
        """
        Call the logs method at the warning level and increment the log index to make an unique id for each log
        """
        self.__index += 1
        self._shared_logger.warning(
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )

    def error(self, *args) -> None:
        """
        Call the logs method at the error level and increment the log index to make an unique id for each log
        """
        self.__index += 1
        self._shared_logger.error(
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )

    def critical(self, *args) -> None:
        """
        Call the logs method at the critical level and increment the log index to make an unique id for each log
        """
        self.__index += 1
        self._shared_logger.critical(
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )


class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add contextual info to log

        Args:
            record: Some information
        Returns:
            The return value is always True
        """
        record.seed = record.args["seed"]
        record.id = record.args["id"]
        return True
