from typing import List, Tuple
from model.tasks.task import State, Task


class TaskSchedulingTrace:
    """
    Class that allows to pair a list of tasks with a list of timestamps,
    such as each task is associated with a time of submission (i.e. it becomes a candidate to execution).

    ATTENTION: this class is just a way to store data.
    """

    def __init__(self, tasks: List[Task], task_submission_timestamps: List[float]):
        """
        Constructor of TaskSchedulingTrace class.
        :param tasks: The list of tasks that simulation shall execute.
        :param task_submission_timestamps: The list of moments at which each task becomes a candidate for execution.
        """
        self.tasks_ts: List[Tuple[Task, float]] = sorted([(task, timestamp) for task, timestamp
                                                          in zip(tasks, task_submission_timestamps)],
                                                         key=lambda t_ple: t_ple[1])
        self.lsft: int = -1    # The index of the "Last Surely Finished Task" in the simulation
        # Be careful, it is not the index of the lastly finished Task,
        # it is such as all tasks before it, in the order of their submission timestamps, are already finished.
        self.lst: int = -1    # The index of the "Last Submitted Task" to simulator's scope.
        # Note that between the indexes of lft and lst, there can be no Task already submitted,
        # because the Tasks are ordered according to their submission timestamps.
        # But there can be some already finished Tasks because their purpose is only to focus the simulation on
        # the most interesting part of the Tasks list, to shorten calculations.

    def update_lsft(self):
        """
        Updates the index of the Last Finished Task.
        :return: None
        """
        lsft = self.lsft
        assert (self.tasks_ts[lsft][0].state is State.FINISHED) or (lsft == -1)
        while lsft < self.lst and self.tasks_ts[lsft+1][0].state is State.FINISHED:
            lsft += 1
        self.lsft = lsft

    def update_lst(self):
        """
        Updates the index of the Last Submitted Task.
        The following task must have already been submitted to the scheduler.
        :return: None
        """
        if self.tasks_ts[self.lst+1][0].state in\
                (State.EXECUTING, State.EXECUTING_IO, State.EXECUTING_CALCULATION, State.QUEUED):
            self.lst += 1
        else:
            raise ValueError(f"The method has been called on a Task that has state "
                             f"'{self.tasks_ts[self.lst+1][0].state}', "
                             f"incompatible with its supposed recent submission.")