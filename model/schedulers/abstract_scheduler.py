from abc import ABC, abstractmethod

from simulation.schedule_order import ScheduleOrder


class AbstractScheduler(ABC):
    """
    Mother class of all schedulers.
    """

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def task_executed(self, schedule_order: ScheduleOrder):
        """
        Performs any modification to be taken into account when Simulation executes a Task.
        :param schedule_order: the ScheduleOrder that just executed.
        :return: None
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

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
