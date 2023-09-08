from typing import List, Tuple

from tools.tasks.task import State, Task


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
        self.lsfti: int = -1  # The "Last Surely Finished Task Index" in the simulation
        # Be careful, it is not the index of the lastly finished Task,
        # it is such as all tasks before it, in the order of their submission timestamps, are already finished.
        self.lsti: int = -1  # The "Last Submitted Task Index" to simulator's scope.
        # Note that between the indexes of lsfti and lsti, all Tasks are already submitted,
        # because the Tasks are ordered according to their submission timestamps.
        # But there can be some already finished Tasks because the purpose of these numbers is only to focus the
        # simulation on the most interesting part of the Tasks list, to shorten calculations.

    def update_lsfti(self):
        """
        Updates the index of the Last Finished Task.
        :return: None
        """
        lsfti = self.lsfti
        assert (self.tasks_ts[lsfti][0].state is State.FINISHED) or (lsfti == -1)
        while (lsfti < self.lsti) and (self.tasks_ts[lsfti + 1][0].state is State.FINISHED):
            lsfti += 1
        self.lsfti = lsfti

    def update_lsti(self):
        """
        Updates the index of the Last Submitted Task.
        The following task must have already been submitted to the scheduler.
        :return: None
        """
        if self.tasks_ts[self.lsti + 1][0].state in \
                (State.SUBMITTED, State.QUEUED, State.EXECUTING, State.EXECUTING_IO, State.EXECUTING_CALCULATION):
            self.lsti += 1
        else:
            raise ValueError(f"The method has been called on a Task that has state "
                             f"'{self.tasks_ts[self.lsti + 1][0].state}', "
                             f"incompatible with its supposed recent submission.")
