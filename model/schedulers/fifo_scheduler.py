from random import Random
from typing import List, Tuple
from colorama import Fore, Style
from model.node import Node
from model.schedulers.abstract_scheduler import AbstractScheduler
from model.storage.abstract_storage_tier import Storage
from model.tasks.task import Task, State


class NaiveScheduler(AbstractScheduler):
    """
    First scheduler to be implemented.
    It simply applies a "First Come, First Served" policy.
    """

    def __init__(self, name: str, env_nodes: List[Node], env_storage: List[Storage], rng: Random):
        """
        Constructor of the NaiveScheduler class.
        :param name: String to name the Scheduler.
        :param env_nodes: List of nodes on the system.
        :param env_storage: List of storage devices on the system.
        :param rng: An instance of a random number generator.
        """
        # List containing all the available resources :
        self.name: str = name   # To create a folder with records.
        self.nodes: List[Node] = env_nodes
        self.storage: List[Storage] = env_storage
        self.rng = rng
        # Queue of all candidate tasks that cannot be executed yet, because of a lack of available resources :
        self.queue: List[Task] = []
        self.current_time: float = 0.
        self.finished_tasks: List[Task] = []    # The list of all tasks that are already finished.

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

    def sort_least_used_nodes(self):
        """
        Search for all nodes in the system that have, at least, one free core,
            and sorts them according to their number of free cores, in a decreasing order.
        :return: A list containing a list of tuples in which the 1st element is a node and the 2nd is its number of
        available cores, and the total of available cores in the system.
        """
        av_nodes: List[Tuple[Node, int]] = []
        total_av_cores: int = 0
        for node in self.nodes:
            unoccupied = node.core_count - node.busy_cores
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

    def executing_task(self, task: Task, insertion: int = 0) -> List[Tuple[Node, int]] or None:
        """
        Check if there are enough resources to execute the task in argument.
        - If yes, try to allocate all the resources needed for it and return the list of nodes & cores to allocate.
        - If no, put the task at the end of the queue and return None.
        :param task: The task to execute
        :param insertion: an int saying, in case the execution of the Task fails, where to insert it in the queue.
        :return: The list of couples (node, cores) to be allocated to the task, or None if the task cannot be executed.
        """
        m_cores = task.min_thread_count  # We assume that one thread occupies exactly one core.
        [av_nodes, tot_av_cores] = self.sort_least_used_nodes()
        if tot_av_cores < m_cores:
            # If there are not enough calcul capacities available, we don't try to execute the application.
            self.queuing(task, insertion=insertion)
            return None
        # We don't check if there is enough I/O bandwidth, because we want to be able to create I/O contention,
        #   and because it depends on each TaskStep, not on each Task.
        reserved_nodes = []
        required_cores = m_cores
        while required_cores:
            remaining_demand = self.plan_node(required_cores, av_nodes[0][1])
            reserved_nodes.append(av_nodes.pop(0))
            # If required_cores > 0, we have to book all available cores of the node.
            # If required_cores = 0, we only book the required number of cores.
            if not remaining_demand:
                reserved_nodes[-1] = (reserved_nodes[-1][0], required_cores)
            required_cores = remaining_demand
        allocated_cores = sum(cores[1] for cores in reserved_nodes)
        assert allocated_cores == m_cores
        return reserved_nodes

    def on_new_task(self, task: Task):
        """
        This algorithm is a FCFS, so when a new task arrives,
            we first check if an earlier task shall be executed.
        - If yes, the new task is put at the end of the queue and nothing else happens
            (because the task that has priority stops the others).
        - If not, it means that the input task has the right of way, so we try to execute it.
        :param task: The oncoming task.
        :return: None
        """
        exe: bool = self.all_dependencies_check(task)
        if len(self.queue) > 0 or not exe:
            self.queuing(task, insertion=-1)
        else:
            resources = self.executing_task(task, insertion=-1)
            if resources:
                task.on_start(resources, self.current_time)

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
        # insort(self.finished_tasks, task, key=lambda the_task: the_task.name)
        self.finished_tasks.append(task)
        print(Fore.RED + repr(self.finished_tasks[-1]) + Style.RESET_ALL)
        if len(self.queue) > 0:
            oncoming_task: Task = self.queue.pop(0)
            deps_check: bool = self.all_dependencies_check(oncoming_task)
            new_order = self.executing_task(oncoming_task, insertion=0)
            if (not new_order) or (not deps_check):
                if new_order and not deps_check:
                    self.queuing(oncoming_task, insertion=0)
                return None
            while new_order and deps_check:
                oncoming_task.on_start(new_order, self.current_time)
                if not self.queue:
                    print(Fore.LIGHTBLUE_EX + f"Queue empty." + Style.RESET_ALL)
                    break
                else:
                    oncoming_task: Task = self.queue.pop(0)
                    deps_check: bool = self.all_dependencies_check(oncoming_task)
                    new_order = self.executing_task(oncoming_task, insertion=0)
            if new_order and not deps_check:
                self.queuing(oncoming_task, insertion=0)
