# from random import Random
from typing import List, Tuple, Set, Dict
from colorama import Fore, Style

from model.node import Node
from model.schedulers.abstract_scheduler import AbstractScheduler
from model.storage.abstract_storage_tier import Storage
from model.tasks.task import Task, State
from simulation.schedule_order import ScheduleOrder, Order


class NaiveScheduler(AbstractScheduler):
    """
    First scheduler to be implemented.
    It simply applies a "First Come, First Served" policy.
    """

    def __init__(self, name: str, env_nodes: List[Node], env_storage: List[Storage]):
        """
        Constructor of the NaiveScheduler class.
        :param name: String to name the Scheduler.
        :param env_nodes: List of nodes on the system.
        :param env_storage: List of storage devices on the system.
        """
        # List containing all the available resources :
        self.name: str = name   # To create a folder with records.
        self.nodes: List[Node] = env_nodes
        self.storage: List[Storage] = env_storage
        # Queue of all candidate tasks that cannot be executed yet, because of a lack of available resources :
        self.queue: List[Task] = []
        self.current_time: float = 0.
        self.finished_tasks: List[Task] = []    # The list of all tasks that are already finished.
        self.pre_allocated_cores: Dict[Node: Dict[Task: int]] = {}  # When a Task required by many others finishes,
        # these many Task will generate ScheduleOrders, making some cores 'pre-allocated'. This dictionary allows the
        # Scheduler to remember which cores should be considered occupied before receiving NextEvents.
        for node in env_nodes:
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

    def sort_least_used_nodes(self, task: Task):
        """
        Search for all nodes in the system that have, at least, one free core,
            and sorts them according to their number of free cores, in a decreasing order.
        :param task: The Task we want to give resources to.
        :return: A list containing a list of tuples in which the 1st element is a node and the 2nd is its number of
        available cores, and the total of available cores in the system.
        """
        av_nodes: List[Tuple[Node, int]] = []
        total_av_cores: int = 0
        for node in self.nodes:
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

    # noinspection PyMethodMayBeStatic
    def plan_node(self, required_cores: int, available_cores: int) -> int:
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

    def all_dependencies_check(self, task: Task) -> bool:
        """
        Check if all Tasks for which completion was required by another one have eventually been satisfied or not.
        :param task: The Task that requires the completion of others.
        :return: True if all dependencies are done, False otherwise.
        """
        deps: List[Task] = task.dependencies
        k: int = 0
        exe: bool = len(deps) <= len(self.finished_tasks)
        while exe and k < len(deps):
            if deps[k] not in self.finished_tasks:
                exe = False
            k += 1
        return exe

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
        # We don't check if there is enough I/O bandwidth, because we want to be able to create I/O contention,
        #   and because it depends on each TaskStep, not on each Task.
        reserved_nodes: List[Tuple[Node, int]] = []
        required_cores = m_cores
        while required_cores:
            remaining_demand = self.plan_node(required_cores, av_nodes[0][1])
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

    def task_executed(self, schedule_order: ScheduleOrder):
        """
        Performs any modification to be taken into account when Simulation executes a Task.
        :param schedule_order: the ScheduleOrder that just executed.
        :return: None
        """
        assert self.current_time == schedule_order.time
        for (node, cores) in schedule_order.task.allocated_cores:
            self.pre_allocated_cores[node][schedule_order.task] -= cores

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
                                              time=self.current_time,
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
        task.on_finish()
        self.finished_tasks.append(task)
        print(Fore.RED + repr(self.finished_tasks[-1]) + Style.RESET_ALL)
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
                                             time=self.current_time,
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
