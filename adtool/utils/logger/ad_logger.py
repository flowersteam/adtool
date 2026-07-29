from __future__ import annotations

import logging
from random import seed
from typing import List


class AutoDiscLogger(logging.Logger):
    """
    A logger to manage experiment logs and print them to configured handlers.
    """

    _LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(
        self,
        experiment_id: int,
        seed: int,
        handlers: List[AutoDiscLogger],
        level: str = "INFO",
    ) -> None:
        """
        Init the logger for an experiment.

        Args:
            experiment_id: current experiment id
            seed: current seed number
            handlers: handlers used to manage experiment logs.
        """
        self.__experiment_id = experiment_id
        self._seed = seed
        self._shared_logger = logging.getLogger("ad_tool_logger")
        self.set_level(level)
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

    def set_level(self, level: str) -> None:
        """Set the minimum emitted level for this experiment process."""
        normalized = str(level).upper()
        if normalized not in self._LEVELS:
            allowed = ", ".join(self._LEVELS)
            raise ValueError(f"Unsupported log level {level!r}. Choose one of: {allowed}.")
        self._shared_logger.setLevel(self._LEVELS[normalized])

    def _emit(self, level: int, *args) -> None:
        """Emit one contextualized record only when its level is enabled."""
        if not self._shared_logger.isEnabledFor(level):
            return
        self.__index += 1
        self._shared_logger.log(
            level,
            *args,
            {
                "experiment_id": self.__experiment_id,
                "seed": self._seed,
                "id": "{}_{}_{}".format(self.__experiment_id, self._seed, self.__index),
            },
        )

    def debug(self, *args) -> None:
        """
        Call the logs method at the debug level and increment the log index to make an unique id for each log
        """
        self._emit(logging.DEBUG, *args)

    def info(self, *args) -> None:
        """
        Call the logs method at the info level and increment the log index to make an unique id for each log
        """
        self._emit(logging.INFO, *args)

    def warning(self, *args) -> None:
        """
        Call the logs method at the warning level and increment the log index to make an unique id for each log
        """
        self._emit(logging.WARNING, *args)

    def error(self, *args) -> None:
        """
        Call the logs method at the error level and increment the log index to make an unique id for each log
        """
        self._emit(logging.ERROR, *args)

    def critical(self, *args) -> None:
        """
        Call the logs method at the critical level and increment the log index to make an unique id for each log
        """
        self._emit(logging.CRITICAL, *args)


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
