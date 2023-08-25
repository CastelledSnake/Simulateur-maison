from abc import ABC, abstractmethod
from enum import auto, Enum


class Event(Enum):
    """
    Describe an event.
    """

    SIMULATION_START = auto()   # The Simulation just began.
    TASK_SUBMIT = auto()    # A Task is to be coming to Simulation's view.
    CALC_STEP_BEGIN = auto()    # A Task is to begin a ComputeTaskStep
    IO_STEP_BEGIN = auto()  # A Task is to begin an IOTaskStep
    CALC_STEP_END = auto()  # A Task is to end a ComputeTaskStep
    IO_STEP_END = auto()    # A Task is to end an IOTaskStep
    TASK_TERMINAISON = auto()   # A Task is about to end (i.e. its last TaskStep is to end).
    SIMULATION_TERMINAISON = auto()     # The Simulation is over, because of no remaining Task to execute.


class AbstractScheduler(ABC):
    """
    Mother class of all schedulers.
    """

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def on_new_task(self, task: "Task"):
        """
        Deals with the arrival of a new task in the queue of candidates.
        :param task: The oncoming task
        :return: A list of schedule orders
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def on_task_finished(self, task: "Task"):
        """
        Deals with the ending of a task.
        :param task: The task that just finished.
        :return: A list of schedule orders
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")