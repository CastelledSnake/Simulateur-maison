# from random import Random
from typing import List, Tuple, Set, Dict, Union
from colorama import Fore, Style
from types import NoneType

from simulation.schedule_order import ScheduleOrder, Order
from simulation.model import Model
from tools.node import Node
from tools.schedulers.abstract_scheduler import AbstractScheduler
# from tools.storage.abstract_storage_tier import Storage
from tools.tasks.task import Task, State


class NaiveScheduler(AbstractScheduler):
    """
    First scheduler to be implemented.
    It simply applies a "First Come, First Served" policy.
    """

    def __init__(self, name: str, model: Model, scoring_function=None):
        """
        Constructor of the NaiveScheduler class.
        :param name: String to name the Scheduler.
        :param model:
        :param scoring_function:
        """
        # List containing all the available resources :
        AbstractScheduler.__init__(self, name, model, scoring_function)  # This scheduler does nothing about
        # scoring_function : it simply tries to execute Tasks as possible, while keeping their order of arrival.
        # Queue of all candidate tasks that cannot be executed yet, because of a lack of available resources :
        self.queue: List[Task] = []
        self.pre_allocated_cores: Dict[Node: Dict[Union[Task, NoneType]: int]] = {}  # When a Task required by many
        # others finishes, Scheduler will generate ScheduleOrders on these Tasks, making some cores 'pre-allocated'.
        # This dictionary allows the Scheduler to remember which cores should be considered occupied before receiving
        # NextEvents.
        for node in self.model.nodes:
            self.pre_allocated_cores[node] = {None: 0}

    def queuing(self, task: Task, insertion=-1) -> None:
        """
        Performs the required operations to put an oncoming task at the end of the queue.
        :param task: The task to put in the queue.
        :param insertion: An int saying, in case the execution of the Task fails, where to insert it in the queue.
        :return: None
        """
        if insertion < 0:
            insertion = len(self.queue) + insertion + 1
        self.queue.insert(insertion, task)
        task.state = State.QUEUED
        for node in self.pre_allocated_cores:
            if task in self.pre_allocated_cores[node]:
                del self.pre_allocated_cores[node][task]
        print(Fore.LIGHTBLUE_EX + f"Queuing {task}" + Style.RESET_ALL)
        print(Fore.LIGHTBLUE_EX + f"Current queue: {self.queue}" + Style.RESET_ALL)

    def find_resources(self, task: Task, insertion: int = 0) -> List[Tuple[Node, int]] or None:
        """
        Check if there are enough resources to execute the task in argument.
        - If yes, try to allocate all the resources needed for it and return the list of nodes & cores to allocate.
        - If no, put the task at the end of the queue and return None.
        :param task: The task to execute
        :param insertion: an int saying, in case the execution of the Task fails, where to insert it in the queue.
        :return: The list of couples (node, cores) to be allocated to the task, or None if the task cannot be executed.
        """
        m_cores = task.min_thread_count  # We assume that one thread occupies exactly one core.
        [av_nodes, tot_av_cores] = self.sort_least_used_nodes(task)
        if tot_av_cores < m_cores:
            # If there are not enough calcul capacities available, we don't try to execute the application.
            self.queuing(task, insertion=insertion)
            return None
        # We don't check if there is enough I/O bandwidth, because we want to be able to create I/O contention.
        return self.book_least_used_cores(task, av_nodes)

    def on_new_task(self, task: Task):
        """
        This algorithm is a FCFS, so when a new task arrives,
            we first check if an earlier task shall be executed.
        - If yes, the new task is put at the end of the queue and nothing else happens
            (because the task that has priority stops the others).
        - If not, it means that the input task has the right of way, so we try to execute it.
        :param task: The oncoming Task.
        :return: None
        """
        exe: bool = self.all_dependencies_check(task)
        start_order: Set[ScheduleOrder] = set()
        if len(self.queue) > 0 or not exe:
            self.queuing(task, insertion=-1)
        else:
            resources = self.find_resources(task, insertion=-1)
            if resources:
                start_order.add(ScheduleOrder(order=Order.START_TASK,
                                              time=self.model.time[-1],
                                              task=task,
                                              nodes=resources))
        return start_order

    def on_task_finished(self, task: Task):
        """
        This algorithm is a FCFS, so when some resource is liberated,
            it first tries to execute the first task in the queue.
        - If it succeeds, it tries the same with the next task in the queue on the remaining resources.
        - If this fails, nothing happens,
            independently of the fact that another task, later in the queue, may be executed.
        :param task: The task that just completed
        :return: None
        """
        new_orders: Set[ScheduleOrder] = set()
        if len(self.queue) > 0:
            oncoming_task: Task = self.queue.pop(0)
            deps_check: bool = self.all_dependencies_check(oncoming_task)
            resources_to_get = self.find_resources(oncoming_task, insertion=0)
            if (not resources_to_get) or (not deps_check):
                if resources_to_get and not deps_check:
                    self.queuing(oncoming_task, insertion=0)
                return new_orders
            while resources_to_get and deps_check:
                new_orders.add(ScheduleOrder(order=Order.START_TASK,
                                             time=self.model.time[-1],
                                             task=oncoming_task,
                                             nodes=resources_to_get))
                if not self.queue:
                    print(Fore.LIGHTBLUE_EX + f"Queue empty." + Style.RESET_ALL)
                    break
                else:
                    oncoming_task: Task = self.queue.pop(0)
                    deps_check = self.all_dependencies_check(oncoming_task)
                    resources_to_get = self.find_resources(oncoming_task, insertion=0)
            if resources_to_get and not deps_check:
                self.queuing(oncoming_task, insertion=0)
        return new_orders
