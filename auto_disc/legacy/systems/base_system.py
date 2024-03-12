from copy import deepcopy

from auto_disc.legacy import BaseAutoDiscModule
from auto_disc.legacy.utils.spaces import DictSpace


class BaseSystem(BaseAutoDiscModule):
    """The main BaseSystem class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        reset
        step
        observe
        render
        close
    The config must be set through ConfigParameters decorators above the system's definition:
    `@IntegerConfigParameter(name="parameter_name", default=100, min=1, max=1000)`
    These parameters can then be accessed through `self.config.parameter_name`
    """

    input_space = DictSpace()
    output_space = DictSpace()
    step_output_space = DictSpace()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_space = deepcopy(self.input_space)
        self.input_space.initialize(self)

        self.output_space = deepcopy(self.output_space)
        self.output_space.initialize(self)

        self.step_output_space = deepcopy(self.step_output_space)
        self.step_output_space.initialize(self)

    def reset(self, run_parameters):
        """
        Resets the environment to an initial state and returns an initial observation.
        Args:
            run_parameters (AttrDict): the input parameters of the system provided by the agent
        Returns:
            observation (AttrDict): the initial observation.
        """
        raise NotImplementedError

    def step(self, action=None):
        """Run one timestep of the system's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (AttrDict): an action provided by the agent
        Returns:
            observation (AttrDict): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (AttrDict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        raise NotImplementedError

    def observe(self):
        """
        Returns the overall output of the system according to the last `reset()`.
        Use this function once the step function has returned `done=True` to give the system's output to the Output Representation (and then the explorer).
        """
        raise NotImplementedError

    def render(self, **kwargs):
        """Renders the environment."""
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass
