from abc import ABC, abstractmethod
from enum import Enum, auto


class TaskStep(ABC):
    """
    Mother class of ComputeTaskStep and IOTaskStep.
    """
    def __init__(self, task: "Task" = None):
        self.task: "Task" = task

    @abstractmethod
    def on_start(self, current_time: float):
        """
        Ask for I/O throughput to be dedicated to the task step.
        :param current_time: The current time at which the TaskStep effectively starts.
        :return: None
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def on_finish(self):
        """
        Free storage resources
        :return: None
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def predict_finish_time(self):
        """
        Computes an estimation of the remaining time to complete the step,
        considering resources allocated and assuming there are no perturbations incoming in the system.
        :return: The estimated remaining time in seconds (float)
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def increment_progress(self, current_time: int):
        """
        Computes the current progression of the task step.
        :return: None
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")
