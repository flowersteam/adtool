import threading
from ast import Raise
from time import sleep
from typing import Any

from adtool.utils.callbacks import BaseCallback

Object = lambda **kwargs: type("Object", (), kwargs)()
global Interact
Interact = None  # Object(callbacks={}, config={"idx":0})


class Interaction:
    def __init__(self) -> None:
        if Interact == None:
            self.interact = {}

    def init_seed(
        self,
        interact_callbacks,
        config,
    ):
        self.interact[threading.current_thread().ident] = {}
        self.interact[threading.current_thread().ident]["callbacks"] = {}
        self.interact[threading.current_thread().ident]["config"] = {}
        self.interact[threading.current_thread().ident]["callbacks"].update(
            interact_callbacks
        )
        self.interact[threading.current_thread().ident]["config"].update(config)

        if "saveExpeDB" in self.interact[threading.current_thread().ident]["callbacks"]:
            self.interact[threading.current_thread().ident][
                "keyDefaultSave"
            ] = "saveExpeDB"
            # self.interactMethod(data, dict_info)
        elif "saveDisk" in self.interact[threading.current_thread().ident]["callbacks"]:
            self.interact[threading.current_thread().ident][
                "keyDefaultSave"
            ] = "saveDisk"
        else:
            raise Exception("No default save interact callback")

        if "readExpeDB" in self.interact[threading.current_thread().ident]["callbacks"]:
            self.interact[threading.current_thread().ident][
                "keyDefaultRead"
            ] = "readExpeDB"
        elif "readDisk" in self.interact[threading.current_thread().ident]["callbacks"]:
            self.interact[threading.current_thread().ident][
                "keyDefaultRead"
            ] = "readDisk"
        else:
            raise Exception("No default read interact callback")

    def __call__(self, callback_name, data, dict_info=None) -> Any:
        self.interact[threading.current_thread().ident]["config"]["idx"] += 1
        return self.interact[threading.current_thread().ident]["callbacks"][
            callback_name
        ](
            data,
            dict_info=dict(
                self.interact[threading.current_thread().ident]["config"], **dict_info
            ),
        )

    def save(self, data, dict_info=None):
        return self.__call__(
            self.interact[threading.current_thread().ident]["keyDefaultSave"],
            data,
            dict_info=dict_info,
        )

    def read(self, filter_attribut=None):
        config = Interact.interact[threading.current_thread().ident]["config"]
        if isinstance(filter_attribut, dict):
            config.update(filter_attribut)

        response = []
        timer = 1
        while response == [] or response == None:
            response = self.interact[threading.current_thread().ident]["callbacks"][
                self.interact[threading.current_thread().ident]["keyDefaultRead"]
            ](config)
            sleep(timer)
            if timer < 64:
                timer *= 2
        return response

    def feedback(self, data, dict_info=None, dict_value_to_change=None):
        if isinstance(dict_info, dict):
            dict_info.update({"type": "question"})
        else:
            dict_info = {"type": "question"}
        self.save(data, dict_info=dict(dict_info, **dict_value_to_change))

        dict_info.update({"type": "answer"})
        return self.read(filter_attribut=dict_info)


Interact = Interaction()


class BaseInteractCallback(BaseCallback):
    """
    Base class for cancelled callbacks used by the experiment pipelines when the experiment was cancelled.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all cancelled callbacks

        Args:
            kwargs: some usefull args (e.g. experiment_id...)
        """
        super().__init__(**kwargs)
        self.interactMethod = kwargs["interactMethod"]

    def __call__(self, data, config, dict_info=None, **kwargs) -> None:
        """
        The function to call to effectively raise cancelled callback.
        Inform the user that the experience is in canceled status
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        self.interactMethod(data, config["seed"], dict_info)
        config["idx"] += 1
