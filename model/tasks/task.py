from enum import Enum, auto
from typing import List, Tuple


class State(Enum):
    """
    Describes the possible states for tasks, from their creation (out of simulation range) to their terminaison.
    """

    NOT_SUBMITTED = auto()  # The task has just been defined: the simulated system hasn't heard of it yet.
    QUEUED = auto()  # The task has been submitted to the scheduler, but no resource is allocated for it.
    EXECUTING = auto()  # The task has been allocated computation resources, but isn't in a calculation or I/O phase
    EXECUTING_CALCULATION = auto()  # The task is performing some calculations.
    EXECUTING_IO = auto()  # The task is performing some I/O.
    FINISHED = auto()  # The task is completed: terminated.


class Task:
    """
    Describes a task
    """

    def __init__(self, name: str, steps: List["TaskStep"], min_thread_count: int = 1,
                 dependencies: List["Task"] = None):
        """
        Constructor of Task class.
        :param name: The name of the task (str).
        :param min_thread_count: The minimum number (int) of threads required to perform the task. Note that we assume
        that one thread takes exactly all the compute resources of one core.
        :param dependencies: List of tasks that must be completed before we can launch that one (List[Task] or None).
        """
        self.name: str = name
        self.min_thread_count: int = min_thread_count
        self.state: State = State.NOT_SUBMITTED
        self.dependencies: List["Task"] = dependencies or list()
        self.steps: List["TaskStep"] = steps
        self.current_step_index: int = -1
        self.allocated_cores: List[Tuple["Node", int]] = []
        # We check that all TaskSteps haven't been yet associated before assigning them to the current Task.
        assert all([step.task is None for step in steps])
        for step in self.steps:
            step.task = self

    def __str__(self):
        # The default method to print tasks if we want to print a lot of them.
        return f'Task "{self.name}"'

    def __repr__(self):
        # Debug version of previous __str__: more detailed, but more verbose.
        human_readable_csi = self.current_step_index + 1
        if not self.dependencies:
            return f'Task "{self.name}", {self.state}, requires {self.min_thread_count} threads, ' \
                   f'is at step {human_readable_csi}/{len(self.steps)}'
        dependencies = list(map(lambda task: task.name, self.dependencies))
        return f'Task "{self.name}", {self.state}, requires {self.min_thread_count} threads, ' \
               f'is at step {human_readable_csi}/{len(self.steps)}, ' \
               f'and must come after tasks: {dependencies}'

    def on_start(self, list_nodes: List[Tuple["Node", int]], current_time: float):
        """
        Reserves computation resources and initiates the TaskStep sequence.
        The Scheduler must have decided all the parameters when calling this method.
        :param list_nodes: A list of tuples, each containing one node on which task shall execute,
        and the number of cores to reserve on this node.
        :param current_time: The current time at which the task effectively starts.
        :return: None
        """
        # The task can be executed without having been put in the queue before.
        assert self.state is State.QUEUED or State.NOT_SUBMITTED
        assert self.current_step_index == -1
        assert all([dep.state is State.FINISHED for dep in self.dependencies])

        self.current_step_index = 0
        self.state = State.EXECUTING
        alloc_cores = 0
        for (node, core_count) in list_nodes:
            self.allocated_cores.append((node, core_count))
            node.register(self, core_count)
            alloc_cores += core_count
        assert alloc_cores >= self.min_thread_count
        self.steps[0].on_start(current_time)

    def on_finish(self):
        """
        Put the task in FINISHED state, terminate the last step and liberates compute resources
        :return: None
        """
        assert self.current_step_index == len(self.steps)-1
        self.steps[-1].on_finish()
        assert self.state is State.EXECUTING
        for (node, core_count) in self.allocated_cores:
            node.unregister(self, core_count)
        self.state = State.FINISHED
