from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
from types import NoneType, FunctionType

from simulation.schedule_order import ScheduleOrder
from simulation.model import Model
from tools.tasks.task import State, Task
from tools.node import Node


def plan_node(required_cores: int, available_cores: int) -> int:
    """
    Tell how many resources on a compute Node must be given to a Task.
    :param required_cores: Number of cores asked by the task.
    :param available_cores: Number of cores available on the node.
    :return: The number of cores that the task asks for, but which couldn't be satisfied because the node if full.
        If all task's expectations are satisfied, return 0.
    """
    assert required_cores >= 0
    assert available_cores >= 0
    if required_cores <= available_cores:
        return 0
    else:
        return required_cores - available_cores


class AbstractScheduler(ABC):
    """
    Mother class of all schedulers.
    """

    def __init__(self, name: str, model: Model, scoring_function=None):
        """

        :param name:
        :param model:
        :param scoring_function:
        """
        self.name: str = name  # To create a folder with records.
        self.model = model
        self.pre_allocated_cores: Dict[Node: Dict[Union[Task, NoneType]: int]] = {}  # When a Task required by many
        # others finishes, Scheduler will generate ScheduleOrders on these Tasks, making some cores 'pre-allocated'.
        # This dictionary allows the Scheduler to remember which cores should be considered occupied before receiving
        # NextEvents.
        for node in self.model.nodes:
            self.pre_allocated_cores[node] = {None: 0}
        self.scoring_function = scoring_function

    def __str__(self):
        return self.__class__.__name__

    def all_dependencies_check(self, task: Task, light_model: Model = None) -> bool:
        """
        Check if all Tasks for which completion was required by another one have eventually been satisfied or not.
        :param task: The Task that requires the completion of others.
        :param light_model: If the Scheduler wants to
        :return: True if all dependencies are done, False otherwise.
        """
        if light_model:
            model = light_model
        else:
            model = self.model
        deps: List[Task] = task.dependencies
        exe: bool = True
        for task_t in model.tasks_trace.tasks_ts:
            if (task_t[0] in deps) and (task_t[0].state != State.FINISHED):
                exe = False
            if not exe:
                break
        return exe

    def sort_least_used_nodes(self, task: Task) -> [List[Tuple[Node, int]], int]:
        """
        Search for all nodes in the system that have, at least, one free core,
            and sorts them according to their number of free cores, in a decreasing order.
        :param task: The Task we want to give resources to.
        :return: A list containing a list of tuples in which the 1st element is a node and the 2nd is its number of
        available cores, and the total of available cores in the system.
        """
        av_nodes: List[Tuple[Node, int]] = []
        total_av_cores: int = 0
        for node in self.model.nodes:
            unoccupied = node.core_count - node.busy_cores - sum(self.pre_allocated_cores[node].values())
            if task in self.pre_allocated_cores[node]:  # If a Task is already planned on the disk, we must not take
                # the cores it will have as unavailable ones.
                unoccupied += self.pre_allocated_cores[node][task]
            assert unoccupied >= 0
            if unoccupied > 0:
                total_av_cores += unoccupied
                av_nodes.append((node, unoccupied))
        av_nodes.sort(key=lambda liste: liste[1], reverse=True)
        return [av_nodes, total_av_cores]

    def book_least_used_cores(self, task: Task, av_nodes: [List[Tuple[Node, int]], int]) -> List[Tuple[Node, int]]:
        """
        Books some cores on some Node(s) to a Task
        :param task: The Task that requires cores.
        :param av_nodes: A list containing a list of tuples in which the 1st element is a node and the 2nd is its
        number of available cores.
        :return: A list of tuples, containing a Node and the number of cores to book on the Node.
        """
        m_cores = task.min_thread_count
        required_cores = m_cores  # We assume that one thread occupies exactly one core.
        reserved_nodes: List[Tuple[Node, int]] = []
        while required_cores:
            remaining_demand = plan_node(required_cores, av_nodes[0][1])
            reserved_nodes.append(av_nodes.pop(0))
            # If required_cores > 0, we have to book all available cores of the node.
            # If required_cores = 0, we only book the required number of cores.
            if not remaining_demand:
                reserved_nodes[-1] = (reserved_nodes[-1][0], required_cores)
            if task not in self.pre_allocated_cores[reserved_nodes[-1][0]]:
                self.pre_allocated_cores[reserved_nodes[-1][0]][task] = required_cores - remaining_demand
            required_cores = remaining_demand
        allocated_cores = sum(cores[1] for cores in reserved_nodes)
        assert allocated_cores == m_cores
        return reserved_nodes

    def task_launched(self, schedule_order: ScheduleOrder):
        """
        Performs any modification to be taken into account when Simulation executes a Task.
        :param schedule_order: the ScheduleOrder that just executed.
        :return: None
        """
        assert self.model.time[-1] >= schedule_order.time  # Ideally, one would assert equality, but the delayed orders
        # make this unavailable.
        for (node, cores) in schedule_order.task.allocated_cores:
            self.pre_allocated_cores[node][schedule_order.task] -= cores

    @abstractmethod
    def find_resources(self, task: Task):
        """
        Check if there are enough resources to execute the task in argument.
        - If yes, try to allocate all the resources needed for it and return the list of nodes & cores to allocate.
        - If no, put the task at the end of the queue and return None.
        :param task: The task to execute
        :return: The list of couples (node, cores) to be allocated to the task, or None if the task cannot be executed.
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
