# from collections import deque, defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Set
from random import Random, sample
from enum import Enum, auto
from matplotlib import pyplot as plt
from numpy import gradient, inf, log10, floor
# from bisect import insort
from colorama import Back, Fore, Style
from os import makedirs
from os.path import isdir
from csv import writer  # , reader


def pp_value(value: int or float, unit: str, round_size: int = 3):
    # TODO : recoder le simulateur avec des entiers uniquement, de précision jusqu'à 10**-30 par rapport aux unités.
    """
    Pretty-prints a value with adequate unit, multiple and size.
    :param value: The numerical value to be printed (int of float)
    :param unit: The physical unit of value (str)
    :param round_size: The limit to round the number to print
    :return: A nice string of the value with its correct multiple and unit (str)
    """
    if value >= 1e30:       # Quetta
        return f'{round(value / 1e30, round_size)} Q{unit}'
    elif value >= 1e27:     # Ronna
        return f'{round(value / 1e27, round_size)} R{unit}'
    elif value >= 1e24:     # Yotta
        return f'{round(value / 1e24, round_size)} Y{unit}'
    elif value >= 1e21:     # Zetta
        return f'{round(value / 1e21, round_size)} Z{unit}'
    elif value >= 1e18:     # Exa
        return f'{round(value / 1e18, round_size)} E{unit}'
    elif value >= 1e15:     # Péta
        return f'{round(value / 1e15, round_size)} P{unit}'
    elif value >= 1e12:     # Téra
        return f'{round(value / 1e12, round_size)} T{unit}'
    elif value >= 1e9:      # Giga
        return f'{round(value / 1e9, round_size)} G{unit}'
    elif value >= 1e6:      # Méga
        return f'{round(value / 1e6, round_size)} M{unit}'
    elif value >= 1e3:      # kilo
        return f'{round(value / 1e3, round_size)} k{unit}'
    elif value >= 1e2:      # hecto
        return f'{round(value / 1e2, round_size)} h{unit}'
    elif value >= 1e1:      # déca
        return f'{round(value / 1e1, round_size)} da{unit}'
    elif value >= 1 or value == 0:  # aucun préfixe
        return f'{value} {unit}'
    elif value >= 1e-1:     # déci
        return f'{round(value * 1e1, round_size)} d{unit}'
    elif value >= 1e-2:     # centi
        return f'{round(value * 1e3, round_size)} c{unit}'
    elif value >= 1e-3:     # milli
        return f'{round(value * 1e3, round_size)} m{unit}'
    elif value >= 1e-6:     # micro
        return f'{round(value * 1e6, round_size)} µ{unit}'
    elif value >= 1e-9:     # nano
        return f'{round(value * 1e9, round_size)} n{unit}'
    elif value >= 1e-12:    # pico
        return f'{round(value * 1e12, round_size)} p{unit}'
    elif value >= 1e-15:    # femto
        return f'{round(value * 1e15, round_size)} f{unit}'
    elif value >= 1e-18:    # atto
        return f'{round(value * 1e18, round_size)} a{unit}'
    elif value >= 1e-21:    # zepto
        return f'{round(value * 1e21, round_size)} z{unit}'
    elif value >= 1e-24:    # yocto
        return f'{round(value * 1e24, round_size)} y{unit}'
    elif value >= 1e-27:    # ronto
        return f'{round(value * 1e27, round_size)} r{unit}'
    if abs(value) > 1e-30:  # quecto
        return f'{round(value * 1e30, round_size)} q{unit}'
    else:
        raise ValueError(f"{value} is absolutely lower than 10**-30.")


def print_names_from_list(liste: List) -> str:
    """
    Print a list of special items (here, Tasks) such as only the "name" field of each
    element is printed, and not the entire __str__ version of it.
    :param liste: A list of elements such as each of these elements has a field "name".
    :return: A string nicely printing each element's name.
    """
    string: str = "["
    for k in range(len(liste)):
        string += liste[k].name
        if k < len(liste) - 1:
            string += ", "
    string += "]"
    return string


def bisect_for_lol(sorted_list: List, item: List, position: int):
    """
    returns one insertion point of list 'item' in the sorted list of lists 'l', in order to keep 'l' sorted according
    to the key given as 'position' : the index of the element to be watched for when comparing the lists.
    :param sorted_list: A list of lists supposed sorted according to their element at index 'position'.
    :param item: The list for which we should find the appropriate position in 'l'.
    :param position: The index of the sublists to be taken into account when comparing.
    :return: The appropriate index for 'item' in 'l'.
    """
    if len(sorted_list) == 1:
        if sorted_list[0][position] < item[position]:
            return 1
        else:
            return 0
    n = len(sorted_list)
    if sorted_list[n//2][position] > item[position]:
        print(sorted_list[:n//2])
        return bisect_for_lol(sorted_list[:n//2], item, position)
    elif sorted_list[n//2][position] < item[position]:
        print(sorted_list[n // 2:])
        return bisect_for_lol(sorted_list[n//2:], item, position) + n//2
    else:
        print(sorted_list[n//2])
        return n//2

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


class TaskStepType(Enum):
    """
    Describes the nature of a TaskStep.
    """

    UNDEFINED = auto()  # The step is of generic type (not supposed to be met at any time in the process).
    IO = auto()     # The step is of type IOTaskStep
    COMPUTATION = auto()    # The step is of type ComputeTaskStep


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


class TaskStep(ABC):
    """
    Mother class of ComputeTaskStep and IOTaskStep.
    """

    def __init__(self, task: "Task"):
        """
        Constructor of TaskStep class.
        :param task: The task to which the step belongs.
        """
        self.task: Task = task
        self.progress: int = 0  # The progression of the task's execution.
        self.previous_time: float = 0.  # The time at which the last evaluation of TaskStep's progression occurred.
        self.current_time: float = 0.   # The current time of the Simulation.
        self.step_type: TaskStepType = TaskStepType.UNDEFINED

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
    def finish_time(self):
        """
        Computes an estimation of the remaining time to complete the step,
        considering resources allocated and assuming there are no perturbations incoming in the system.
        :return: The estimated remaining time in seconds (float)
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def increment_progress(self):
        """
        Computes the current progression of the task step.
        :return: None
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")


class ComputeTaskStep(TaskStep):
    """
    Describes a step of a task that is dedicated to computation.
    For this class, progression is measured with an integer (the amount of flop done)
    """

    def __init__(self, task: "Task", flop: int):
        """
        Constructor of ComputeTaskStep class.
        :param task: The task that has the ComputeStep.
        :param flop: The number of floating point operations that requires the step to complete (int).
        """
        TaskStep.__init__(self, task)
        self.flop: int = flop
        self.available_flops: float = 0  # The computation speed allocated to the task.
        self.step_type = TaskStepType.COMPUTATION

    def on_start(self, current_time: float):
        """
        Start the ComputeTaskStep, activate calculation on reserved cores.
        Note that compute resources are given when a task is elected, and that they are not dropped until completion.
        :param current_time: The current time at which the TaskStep effectively starts.
        :return: None
        """
        self.previous_time = current_time
        self.current_time = current_time
        for node_th in self.task.allocated_cores:
            node_th[0].idle_busy_cores -= node_th[1]
        assert self.task.state is State.EXECUTING   # A Task can only execute one TaskStep at a time,
        # so it must execute nothing when launching a ComputeTaskStep.
        self.task.state = State.EXECUTING_CALCULATION

    def on_finish(self):
        """
        End the ComputeTaskStep, deactivate calculation on reserved cores.
        Note that compute resources are given when a task is elected, and that they are not dropped until completion.
        :return: None
        """
        for node_th in self.task.allocated_cores:
            node_th[0].idle_busy_cores += node_th[1]
        assert self.task.state is State.EXECUTING_CALCULATION
        self.task.state = State.EXECUTING

    def finish_time(self):
        """
        Computes an estimation of the remaining time to complete the ComputeTaskStep,
            considering resources allocated and assuming there are no perturbations incoming in the system.
        :return: The estimated remaining time in seconds (float)
        """
        flops = 0
        self.increment_progress()
        for resource in self.task.allocated_cores:
            # Here, resource is a list of a node, and the number of cores we occupy on it.
            flops += resource[0].frequency * resource[1]
        self.available_flops = flops
        assert self.progress <= self.flop
        return (self.flop - self.progress) / flops

    def increment_progress(self):
        """
        Computes the current progression of the ComputeTaskStep.
        :return: None
        """
        self.progress += int((self.current_time - self.previous_time) * self.available_flops)
        assert self.progress >= 0
        self.previous_time = self.current_time  # Update the previous time.
        #   The current time will be updated later, in the simulation level.


class IOTaskStep(TaskStep):
    """
    Describes a step of a task that is dedicated to IO.
    """

    def __init__(self, task: "Task", list_storage: List[Tuple["Storage", float]], total_io_volume: int):
        """
        Constructor of IOTaskStep class.
        :param task: The task that has the IOTaskStep.
        :param list_storage: The list of storage instances that are concerned with IO of this step,
            and the bandwidth (B/s) the task requires on each one.
            The I/O flow is considered as constant for the whole execution.
        :param total_io_volume: The total amount of IO required completing the step (B)
        """
        TaskStep.__init__(self, task)
        self.list_storage: List[Tuple["Storage", float]] = list_storage
        self.total_io_volume: int = total_io_volume
        self.step_type: TaskStepType = TaskStepType.IO
        self.available_bandwidth: float = 0.

    def on_start(self, current_time: float):
        """
        Start the IOTaskStep, ask for I/O throughput to be dedicated to the task step.
        :param current_time: The current time at which the TaskStep effectively starts.
        :return: None
        """
        self.previous_time = current_time
        self.current_time = current_time
        assert self.task.state is State.EXECUTING   # A Task can only execute one TaskStep at a time,
        # so it must execute nothing when launching an IOTaskStep.
        self.task.state = State.EXECUTING_IO
        self.list_storage[0][0].register(self.task, self.list_storage[0][1])    ### REMOVE

    def on_finish(self):
        """
        End the IOTaskStep, liberate storage resources.
        :return: None
        """
        assert self.task.state is State.EXECUTING_IO
        self.list_storage[0][0].unregister(self.task, self.list_storage[0][1])  ### REMOVE
        self.task.state = State.EXECUTING

    def finish_time(self):
        """
        Computes an estimation of the remaining time to complete the IOTaskStep,
            considering resources allocated and assuming there are no perturbations incoming in the system.
        :return: The estimated remaining time in seconds (float)
        """
        bp = 0.
        self.increment_progress()
        for resource in self.list_storage:
            # Here, resource is a storage device and the amount of BP the TaskStep takes on it.
            bp += resource[1] / resource[0].contention
        self.available_bandwidth = bp
        if self.progress > self.total_io_volume:
            progress_error = f"IOTaskStep.progress = {self.progress} > self.total_io_volume = {self.total_io_volume}"
            raise ValueError(progress_error)
        return (self.total_io_volume - self.progress) / self.available_bandwidth  # \
        #    + sum(storage_level[0].latency for storage_level in self.list_storage)
        # TODO Implémenter une meilleure manière de gérer la latence.

    def increment_progress(self):
        """
        Computes the current progression of the IOTaskStep.
        :return: None
        """
        last_progress = int((self.current_time - self.previous_time) * self.available_bandwidth)
        assert last_progress >= 0
        self.progress += last_progress
        self.previous_time = self.current_time  # Update the previous time.
        #   The current time will be updated later, in the simulation level.


class Task:
    """
    Describes a task
    """

    def __init__(self, name: str, min_thread_count: int = 1, dependencies: List["Task"] = None):
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
        if dependencies:
            self.dependencies: List["Task"] = dependencies
        else:
            self.dependencies = []
        self.steps: List[TaskStep] = []
        self.current_step_index: int = -1
        self.allocated_cores: List[Tuple["Node", int]] = []

    def __str__(self):
        # The default method to print tasks if we want to print a lot of them.
        return f'Task "{self.name}"'

    def __repr__(self):
        # Debug version of previous __str__: more detailed, but more verbose.
        human_readable_csi = self.current_step_index + 1
        if not self.dependencies:
            return f'Task "{self.name}", {self.state}, requires {self.min_thread_count} threads, ' \
                   f'is at step {human_readable_csi}/{len(self.steps)}'
        dependencies = print_names_from_list(self.dependencies)
        return f'Task "{self.name}", {self.state}, requires {self.min_thread_count} threads, ' \
               f'is at step {human_readable_csi}/{len(self.steps)}, ' \
               f'and must come after tasks: {dependencies}'

    def add_task_steps(self, steps: List[TaskStep]):
        """
        Add a list of TaskSteps to the Task's field 'self.steps'.
        :param steps: The list of TaskSteps
        :return:
        """
        for task_step in steps:
            if task_step.task == self:
                self.steps.append(task_step)
            else:
                raise AttributeError(f"TaskStep '{task_step}' is not supposed to be part of task '{self}', "
                                     f"but of task '{task_step.task}'")

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
        for dep in self.dependencies:
            if dep.state != State.FINISHED:
                raise ValueError(f"Impossible to launch the task {self}, because dependency {dep} is {dep.state}.")
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


class Node:
    """
    Class describing a compute node.
    """

    def __init__(self, name: str, max_frequency: float, min_frequency: float, core_count: int,
                 static_power: float = 100., sleep_power: float = 0., coefficient_dynamic_power: float = 1e-28,
                 coefficient_leakage_power: float = 1e-9):
        """
        Constructor of Node class.
        :param name: The name of the Node (str).
        :param max_frequency: The maximum frequency that Node's clock can reach (Hz).
        :param min_frequency: The minimum frequency that Node's clock can have (Hz).
        :param core_count: Total number of cores on the Node (int).
        :param static_power: Basis power consumption (W) of the node, it's the minimal power consumption of the Node
        when it's not sleeping.
        :param sleep_power: Power consumption (W) of the node when it's switched off.
        :param coefficient_dynamic_power: A coefficient, inserted in the equation of power consumption before f**3.
        :param coefficient_leakage_power: A coefficient, inserted in the equation of power consumption before f**1.
        """
        self.name: str = name
        self.max_frequency: float = max_frequency
        self.min_frequency: float = min_frequency
        self.frequency: float = max_frequency  # By default, the Node takes the highest frequency it can.
        self.core_count: int = core_count
        self.static_power: float = static_power
        self.sleep_power: float = sleep_power
        self.cdp = coefficient_dynamic_power
        self.clp = coefficient_leakage_power
        # TODO : GROSSE INCOHERENCE : avec k = 1 et frequency = 5e9, on dépasse TRES vite les QJ de puissance consommée
        self.busy_cores: int = 0  # Number of cores running a Task on the node.
        self.idle_busy_cores: int = 0  # Number of cores reserved by some Task, but not performing calculations now.
        # By default, we consider that a Task performs calculation on a Node, iff it is in a ComputationTaskStep.
        self.running_tasks: List[Task] = []  # List of Tasks occupying at least one core on the Node.
        self.sleeping: bool = False   # Boolean saying if the Node is switched off ( = True) or not ( = False).

#    def __setattr__(self, frequency, new_freq: float):
#        if self.min_frequency <= new_freq <= self.max_frequency:
#            self.frequency = new_freq

    def __str__(self):
        return f'Node "{self.name}" running {self.busy_cores}({self.idle_busy_cores})/{self.core_count} ' \
               f'cores for Tasks: {print_names_from_list(self.running_tasks)} ' \
               f'with current frequency = {pp_value(self.frequency, "Hz")}' \
            # f "taking powers: [max = {pp_value(self.busy_core_power, 'W')}, " \
        # f"min = {pp_value(self.idle_core_power, 'W')}, idle = {pp_value(self.sleep_power, 'W')}] "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this Node.
        :return: The power consumption (W) as a float.
        """
        # Plus de formules, plus complexes/complètes ont été vues dans :
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7279063
        # Notamment: Power_CPU = a*f**3 + b*f + c,
        # Telle que bf correspond à la puissance consommée par les caches, et celle résiduelle perdue par les cœurs ;
        # af**3 et c sont resp La puissance dynamique du CPU et la puissance de base, qu'on utilise toutes deux déjà.
        # NB : a et b sont dépendants des applications exécutées, car le cache et la mémoire peuvent avoir des
        # comportements différents suivant les tâches effectuées.
        #
        # Il sera compliqué de trouver plus précis que ce modèle de polynôme de degré 3 sans considérer
        # une hiérarchie dans les cœurs, et/ou la modélisation des caches dans le processeur,
        # et/ou l'inégalité 1 thread != 1 cœur.
        # Il n'est pas prévu d'implémenter des formules à destination des GPUs.
        dynamic_power = self.cdp * (self.busy_cores-self.idle_busy_cores) * self.frequency**3
        leakage_power = self.clp * self.core_count * self.frequency
        return dynamic_power + leakage_power + self.static_power

    def register(self, task: Task, cores: int):
        """
        Register a Task on some cores of the Node.
        It is assumed that the Scheduler has already checked that the Node has sufficient available cores.
        :param task: Newly elected Task for an execution on the Node.
        :param cores: The number of cores to allocate to the Task.
        :return: None
        """
        self.running_tasks.append(task)
        self.busy_cores += cores
        self.idle_busy_cores += cores
        assert 0 <= self.idle_busy_cores <= self.busy_cores <= self.core_count

    def unregister(self, task: Task, cores: int):
        """
        Unregister a Task from all the cores it has on the Node.
        :param task: The task to unregister.
        :param cores: The number of cores the Task has taken on the Node.
        :return: None
        """
        self.running_tasks.remove(task)
        self.busy_cores -= cores
        self.idle_busy_cores -= cores
        assert 0 <= self.idle_busy_cores <= self.busy_cores <= self.core_count


class Storage:
    """
    Mother class of all storage devices implemented (HDD, SSD, RAM, ...).
    """
    # TODO Nouvelle modélisation:
    # On a 2 niveaux de mémoire : HDD et SSD.
    # Le SSD a une latence faible, mais un débit d'IO faible, quand le HDD a le contraire. (cf. nombres d'état de l'art)
    # NB : voir à partir de quelle quantité d'I/O par ordre d'I/O, le HDD devient plus rentable que le SSD.
    #
    # On implémentera la classe "File", dont chaque instance décrit un fichier sur le système.
    # Chaque File a une empreinte mémoire (variable) et un tier de stockage de prédilection.
    # Toute IOTaskStep peut demander un transfert de données à un ou plusieurs File(s), avec un volume total requis,
    #   un débit demandé, un type de transfert : READ ou WRITE, un type d'IO (séquentiel ou en plusieurs parties) et
    # L'ordonnanceur aura la possibilité de transférer un File d'un niveau de stockage à un autre.
    # Chaque support de stockage a ses caractéristiques pour déterminer le temps des I/O, cf. mail de LM.
    # C'est le scheduler qui décide de quoi est ou.
    # La latence est une ordonnée à l'origine : un temps à attendre sans qu'une progression ne se fasse sur les I/O
    #   demandées.
    # On introduit un débit max en sortie de chaque nœud.
    def __init__(self, name: str, capacity: int, throughput: float, latency: float):
        """
        Constructor of the Storage class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        """
        self.name: str = name
        self.capacity: int = capacity
        self.throughput: float = throughput
        self.latency: float = latency
        self.files: Set[File] = set()  # All the files on the Storage will be loaded later on.
        self.occupation: int = 0    # Number of bytes occupied on the Storage.
        self.flow: float = 0.  # Raw amount of I/O-bandwidth (B/s) used at this time.
        self.running_tasks: List[Task] = []
        self.occupied_space: List[List[int, int, "File"]] = []  # A list saying which bytes are taken by which File.
        # It is of the form : [..., [begin_byte_i, space_taken_i, file_i], ...]
        self.empty_space: List[List[int]] = [[0, capacity]]    # A list saying which bytes are freed.
        # It is of the form : [..., [begin_byte_i, space_freed_i], ...]
        self.contention: float = 1.  # A number useful to compute delays linked with I/O-contention
        #   si flow <= throughput : 1 (aucun problème)
        #   si flow > throughput : flow / throughput
        #       (le temps mis pour compléter une quantité d'I/O devient temps_initial * contention,
        #       et la BP réelle disponible pour une tâche devient contention / bp_initiale).

    def __str__(self):
        return f"Storage '{self.name}' " \
               f"with {pp_value(self.occupation, 'B')} out of {pp_value(self.capacity, 'B')} occupied " \
               f"throughput = {pp_value(self.flow, 'B/s')} in {pp_value(self.throughput, 'B/s')} " \
               f"for Tasks: {print_names_from_list(self.running_tasks)} " \
               f"contention = {self.contention} " \
               f"latency = {pp_value(self.latency, 's')} "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this Storage.
        This is only the skeleton of the methodNeeds to be implemented in each subclass.
        :return: A NotImplementedError
        """
        raise NotImplementedError("This is the method from class Storage, Call a method from a subclass.")

    def update_contention(self):
        """
        Determine if the storage device experiences I/O-contention
            and modifies the contention instance variable consequently.
        :return: None
        """
        if self.flow <= self.throughput:
            self.contention = 1.
        else:
            self.contention = self.flow / self.throughput

    def register(self, task: Task, bandwidth: float):
        """
        Give a Task part of the available I/O bandwidth.
        :param task: Newly elected Task for an I/O booking.
        :param bandwidth: The amount of bandwidth to allocate to the Task.
        :return: None
        """
        assert task.steps[task.current_step_index].step_type == TaskStepType.IO
        assert self.flow >= 0
        assert bandwidth >= 0
        self.flow += bandwidth    # Currently, the task_step decide which amount of bp to be taken on each Storage
        self.running_tasks.append(task)
        self.update_contention()

    def unregister(self, task: Task, bandwidth: float):
        """
        Deallocate the I/O bandwidth that a Task has.
        :param task: The Task that liberates resources.
        :param bandwidth: The amount of bandwidth to deallocate from the Task.
        :return: None
        """
        assert task.steps[task.current_step_index].step_type == TaskStepType.IO
        delta = self.flow - bandwidth
        if delta < 0:
            if abs(delta) > 1e-6:   # If the delta is below 0, and lower than 1 µs, the error is probably due to
                # floating point imprecision.
                raise ValueError(f"The I/O flow {self.flow} of Storage {self.name} is about to be lowered of "
                                 f"{bandwidth}")
            else:
                self.flow = 0.
        else:
            self.flow -= bandwidth
        self.running_tasks.remove(task)
        self.update_contention()


class HDDState(Enum):
    """
    Class describing all the states an HDD can take.
    """

    NOMINAL = auto()
    SLOW = auto()
    SLEEP = auto()


class HDD(Storage):
    """
    Class describing a storage device of type Hard Disk Drive: a hard disk always spinning and an arm moving on top of
    the disk to perform reads and writes (as well as the embedded electronics).
    """

    def __init__(self, name: str, capacity: int, throughput: float, latency: float,
                 disk_max_spin: float, disk_min_spin: float,
                 arm_mass: float, arm_radius: float, arm_max_spin: float,
                 electronic_power: float, sleep_power: float = 0.):
        """
        Constructor of the HDD class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        :param disk_max_spin: Maximum angular speed (rad/s) allowed for the disk ().
        :param disk_min_spin: Minimum angular speed (rad/s) allowed for the disk (taken in idle mode).
        :param arm_mass: Mass (kg) of the I/O arm
        :param arm_radius: Radius (m) of the I/O arm
        :param electronic_power: Power consumed by all electronic components of the HDD
        :param sleep_power: power consumed by the HDD when sleeping (default 0)
        """
        super().__init__(name, capacity, throughput, latency)
        self.disk_max_spin: float = disk_max_spin
        self.disk_min_spin: float = disk_min_spin
        self.arm_momentum: float = arm_mass * (2*arm_radius)**2 / 12
        self.arm_radius: float = arm_radius
        self.arm_max_spin: float = arm_max_spin
        self.electronic_power: float = electronic_power
        self.sleep_power = sleep_power
        self.disk_current_spin: float = disk_max_spin  # Let suppose that the HDD begins in nominal mode.
        self.arm_current_spin: float = arm_max_spin    # Let suppose that the HDD begins in nominal mode.
        self.state = HDDState.NOMINAL

    def __str__(self):
        return f"HDD '{self.name}' " \
               f"with {pp_value(self.occupation, 'B')} out of {pp_value(self.capacity, 'B')} occupied " \
               f"throughput = {pp_value(self.flow, 'B/s')} in {pp_value(self.throughput, 'B/s')} " \
               f"for Tasks: {print_names_from_list(self.running_tasks)} contention = {self.contention} " \
               f"currently {self.state} with disk_spin = {pp_value(self.disk_current_spin, 'rad/s')} " \
               f"and arm_speed = {pp_value(self.arm_current_spin, 'rad/s')} currently {self.state} "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this HDD.
        :return: The power consumption (W) as a float.
        This formula comes from this study :
        """
        # Proposition pour HDD :
        #   Réutiliser la formule de l'étude de 1990 : P_SPM = w_SPM**2.8 * (2r)**4.6 (on prend 1 unique disque par HDD)
        #   TODO analyse dimensionnelle.
        #   Pour le VCM : ajouter une formule inspirée d'une autre étude : E_VCM = w_VCM**2 * J_VCM / 2
        #   (cf. cours pour calculer le moment d'inertie)
        #   Pour l'électronique, ajouter un petit offset au total.
        #   Introduire la notion de "repos" : le bras ne tourne plus,
        #   et la "veille" : le disque tourne au ralenti et le bras est immobile.
        if self.state == HDDState.SLEEP:
            return self.sleep_power
        elif self.state == HDDState.SLOW:
            return self.disk_current_spin**2.8 * (2 * self.arm_radius)**4.6 + self.electronic_power
        elif self.state == HDDState.NOMINAL:
            p_spm = self.disk_current_spin**2.8 * (2 * self.arm_radius)**4.6
            p_vcm = self.arm_current_spin ** 2 * self.arm_momentum / 2
            return p_spm + p_vcm + self.electronic_power
        else:
            raise ValueError(f"SSD {self} has state {self.state}.")


class SSDState(Enum):
    """
    Describes the possible states for SSDs.
    """

    ACTIVE = auto()
    SLEEP = auto()


class SSD(Storage):
    """
    Class describing a storage device of type Solid-State Drive: a set of flash memory units that can keep data.
    """

    def __init__(self, name: str, capacity: int, throughput: float, latency: float, max_read_power: float,
                 max_write_power: float, leakage_power: float, sleep_power: float = 0.):
        """
        Constructor of the SSD class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        :param max_read_power: Power consumed if all the bandwidth is used to read.
        :param max_write_power: Power consumed if all the bandwidth is used to write.
        :param leakage_power: Power consumed by the disk for its own needs.
        :param sleep_power: Power consumed by the SSD when sleeping (default 0)
        """
        super().__init__(name, capacity, throughput, latency)
        self.max_read_power: float = max_read_power
        self.max_write_power: float = max_write_power
        self.leakage_power: float = leakage_power
        self.sleep_power: float = sleep_power
        self.flow_read: float = 0.    # Amount of bandwidth (B/s) taken to read data.
        self.flow_write: float = 0.   # Amount of bandwidth (B/s) taken to write data.
        self.state = SSDState.ACTIVE

    def __str__(self):
        return f"SSD '{self.name}' " \
               f"with {pp_value(self.occupation, 'B')} out of {pp_value(self.capacity, 'B')} occupied " \
               f"global throughput = {pp_value(self.flow, 'B/s')} in {pp_value(self.throughput, 'B/s')} " \
               f"of which read = {pp_value(self.flow_read, 'B/s')} " \
               f"and write = {pp_value(self.flow_write, 'B/s')} " \
               f"for Tasks: {print_names_from_list(self.running_tasks)} contention = {self.contention} " \
               f"currently {self.state} "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this SSD.
        :return: The power consumption (W) as a float.
        This formula comes from this study :
        """
        # Proposition pour SSD :
        # Params : les données constructeurs self.max_write_power, self.max_read_power, self.idle_power
        #   Distinguer 2 modes :
        #       Actif : Le disque normal, P = (flow_w/BP)*mwp + (flow_r/BP)*mrp + idle_power
        #       Veille : Le disque ne peut pas lire/écrire, sa consommation vaut self.idle_power
        if self.state == SSDState.ACTIVE:
            return (self.flow_read*self.max_read_power + self.flow_write*self.max_write_power)\
                / self.throughput + self.leakage_power
        elif self.state == SSDState.SLEEP:
            return self.sleep_power
        else:
            raise ValueError(f"SSD {self} has state {self.state}.")


class File:
    """
    Class describing a file on the system. These files can be red or written by IOTaskSteps, allowing to model I/Os.
    """

    def __init__(self, name: str, space: int, preferred_storage: Storage):
        """
        Constructor of the File class.
        :param name: The name of the File instance.
        :param space: The number of bytes the File occupies (this number is mutable).
        :param preferred_storage: The preferred kind of Storage device to put the File on.
        """
        self.name: str = name
        self.space: int = space
        self.preferred_storage = preferred_storage
        self.locations: List[Union[Storage, int, int]] = []  # List used to describe the position of the file.
        self.split_in_disk: bool = False            # Boolean saying iff the file is split between several locations on
        # the same disk. Then self.locations is like : List[..., (disk, first_byte_i, subspace_i), ...]
        # We do not consider the possibility for a File to be split between disks.
        #   self.split_between_disks: bool = False      # Boolean saying iff the file is split between several disks of
        #   the same level. Then self.locations is like : List[..., (disk_i, first_byte_i, subspace_i), ...]
        #   self.split_between_levels: bool = False     # Boolean saying iff the file is split between several disks of
        #   different levels. Then self.locations is like : List[..., (HDD_i, fB_i, sbs_i), (SSD_j, fB_j, sbs_j), ...]
        self.read: bool = False     # Boolean saying iff the file is being red by at least one Task.
        self.written: bool = False  # Boolean saying iff the file is being written by a Task. Only one Task can write a
        # File at a time.

    def __str__(self):
        return f"File '{self.name}', taking {pp_value(self.space, 'B', 3)} " \
               f"on {self.locations} " \
               f"read?-=>{self.read}, written?-=>{self.written} "


class AbstractScheduler(ABC):
    """
    Mother class of all schedulers.
    """

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def on_new_task(self, task: Task):
        """
        Deals with the arrival of a new task in the queue of candidates.
        :param task: The oncoming task
        :return: A list of schedule orders
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")

    @abstractmethod
    def on_task_finished(self, task: Task):
        """
        Deals with the ending of a task.
        :param task: The task that just finished.
        :return: A list of schedule orders
        """
        raise NotImplementedError("The abstract method from the abstract class was called.")


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


class Simulation:
    """
    Class describing the simulation instances: each instance of this class is
        a simulation of the execution of a workflow on hardware (both given as arguments).
    """

    def __init__(self, list_nodes: List[Node], list_storage: [List[HDD], List[SSD]], list_files: List[File],
                 task_list: TaskSchedulingTrace, algo_scheduler):
        """
        Constructor of the Simulation class
        :param list_nodes: The list of all Nodes in the simulated system.
        :param list_storage: The list of all Storage devices in the simulated system.
        :param list_files: The list of all Files initially written on the memory devices.
        :param task_list: The list of all tasks that simulation will have to execute.
        :param algo_scheduler: The scheduling algorithm that will be used in the simulation.
        """
        self.nodes: List[Node] = list_nodes
        self.list_hdds: List[HDD] = list_storage[0]
        self.list_ssds: List[SSD] = list_storage[1]
        self.storage: List[Storage] = list_storage[0] + list_storage[1]
        self.list_files: List[File] = list_files
        self.allocate_files()
        self.task_list: TaskSchedulingTrace = task_list
        self.scheduler = algo_scheduler
        self.next_event: List[Union[List[Event], Task, float]] =\
            [[Event.SIMULATION_START], self.task_list.tasks_ts[0][0], self.task_list.tasks_ts[0][1]]
        self.time: List[float] = [0.]  # The different times of interest raised in the simulation
        self.energy: List[float] = [0.]  # Total energy consumption from the beginning of the simulation
        self.event: int = -1
        # Creation of the folders that will contain the results of the Simulation
        k: int = 1
        while isdir(f"enregistrements_automatiques/{self.scheduler.name}/résultats_{k}"):
            k += 1
        makedirs(f"enregistrements_automatiques/{self.scheduler.name}/résultats_{k}")
        self.record_folder = f"enregistrements_automatiques/{self.scheduler.name}/résultats_{k}"
        makedirs(f"{self.record_folder}/nœuds")
        makedirs(f"{self.record_folder}/mémoire")
        makedirs(f"{self.record_folder}/tâches")
        makedirs(f"{self.record_folder}/global")
        # Creation of the recording files (one per item of interest).
        for node in self.nodes:
            file = open(f"{self.record_folder}/nœuds/{node.name}.csv", "x", newline='')
            writer(file).writerow(["Temps (s)", "Énergie (J)"])
        for storage_unit in self.storage:
            file = open(f"{self.record_folder}/mémoire/{storage_unit.name}.csv", "x", newline='')
            writer(file).writerow(["Temps (s)", "Énergie (J)", "Débit d'E/S (O/s)", "Contention (sans unité)"])
        for task_t in self.task_list.tasks_ts:
            file = open(f"{self.record_folder}/tâches/{task_t[0].name}.csv", "x", newline='')
            writer(file).writerow(["Temps (s)", "Énergie (J)", "Débit d'E/S (O/s)"])
        file = open(f"{self.record_folder}/global/simulation.csv", "x", newline='')
        writer(file).writerow(["Temps (s)", "Énergie (J)"])

    def __str__(self):
        r = f'State of the {self.scheduler} at {self.time[-1]}s, event {self.event} :\nEnergy consumed : ' \
            f'{pp_value(self.energy[-1], "J")}\nNodes available: '
        r += print_list(self.nodes)
        r += '; Storage available : '
        r += print_list(self.storage)
        r += '\n'
        return r

    # noinspection PyMethodMayBeStatic
    def select_attribute_disk(self, file: File, list1: List[Storage], list2: List[Storage]):
        """
        Select a random Storage from the lists given that can welcome the given file.
        :param file: The File to write on a disk.
        :param list1: First set of Storages. First, tries to write the File on one random disk from that list.
        :param list2: Second set of Storages. If the File couldn't be written on one Storage from the 1st list,
        the function tries on disks from that list.
        :return: None
        """
        k: int = 0
        # Rearrange Storage items to ensure that all disks can be used randomly.
        reordered_list_storage: List[Storage] = sample(list1, k=len(list1)) + sample(list2, k=len(list2))
        while k < len(reordered_list_storage):
            if reordered_list_storage[k].throughput - reordered_list_storage[k].occupation >= file.space:
                # The decision on where to write the File on the disk is given to the controller.
                self.scheduler.allocate_space(file, reordered_list_storage[k])
                break
            k += 1
        if k == len(reordered_list_storage):
            raise OverflowError(f"File {file} is too big, and cannot be stored into one disk.")

    def allocate_files(self):
        """
        Performs the computation required to write orphan Files onto some storage (HDD or SSD).
        :return: None
        """
        for file in self.list_files:
            if file.preferred_storage.throughput - file.preferred_storage.occupation >= file.space:
                self.scheduler.allocate_space(file, file.preferred_storage)
            else:
                if file.preferred_storage.name[0:3] == "HDD":
                    self.select_attribute_disk(file, self.list_hdds, self.list_ssds)
                elif file.preferred_storage.name[0:3] == "SSD":
                    self.select_attribute_disk(file, self.list_ssds, self.list_hdds)
                else:
                    self.select_attribute_disk(file, self.list_hdds, self.list_ssds)    # If the name of preferred disk
                    # doesn't follow the pattern and cannot welcome the File, we preferably place the File on an HDD.

    # noinspection PyMethodMayBeStatic
    def nature_of_step_end(self, step_type: State, task: Task, fin_step: float) -> list[list[Event], Task, float]:
        """
        Determines what is the exact kind of TaskStep ending (i.e. the type of the step ending and weather the whole)
        Task is finished or not
        :param step_type: The type of the TaskStep ending.
        :param task: The Task to which the Step belongs.
        :param fin_step: The time at which the Step should end.
        :return: A next_event object.
        """
        # By default, the step's end is not the task's end, because there may be another step following.
        if step_type == State.EXECUTING_IO:
            next_event = [[Event.IO_STEP_END], task, fin_step]
        elif step_type == State.EXECUTING_CALCULATION:
            next_event = [[Event.CALC_STEP_END], task, fin_step]
        else:
            raise ValueError(f"Simulation.step_type has the value '{step_type}', unappropriated for a step_end.")
        if task.current_step_index == len(task.steps) - 1:
            # Here, this is the end of the last step in the task, i.e. the end of the task.
            next_event = [[next_event[0][0], Event.TASK_TERMINAISON], task, fin_step]
        return next_event

    def update_next_event(self):
        """
        Determines what is the next event in the simulation.
        It can be a task completion, a task arrival or the end of the simulation.
        :return: None
        """
        print(Fore.BLUE + f"lst : {self.task_list.lst}, lsft : {self.task_list.lsft}" + Style.RESET_ALL)
        lst = self.task_list.lst
        # Initialization of variable next_event.
        if lst < len(self.task_list.tasks_ts)-1:
            next_event = [[Event.TASK_SUBMIT], self.task_list.tasks_ts[lst+1][0], self.task_list.tasks_ts[lst+1][1]]
        else:
            next_event = [[Event.TASK_SUBMIT], None, inf]
        # Iteration of next_event on all submitted but non-finished tasks.
        for task_t in self.task_list.tasks_ts[self.task_list.lsft+1: lst+1]:     # task_t stands for "task, time"
            if task_t[0].state in [State.EXECUTING_CALCULATION, State.EXECUTING_IO]:
                fin_step = task_t[0].steps[task_t[0].current_step_index].finish_time() + self.time[-1]
                if fin_step <= next_event[2]:
                    next_event = self.nature_of_step_end(task_t[0].state, task_t[0], fin_step)
        # If no task performs any action, it means all tasks have been executed: this is the end of the simulation.
        if next_event[2] == inf:
            next_event = [[Event.SIMULATION_TERMINAISON], None, self.time[-1]]
        self.next_event = next_event

    def alternate_steps(self):
        """
        Performs all te actions required to switch between two TaskSteps.
        :return: None
        """
        task: Task = self.next_event[1]
        task.steps[task.current_step_index].on_finish()
        task.current_step_index += 1
        task.steps[task.current_step_index].on_start(self.time[-1])

    def update_time(self, time: float):
        """
        Add a new moment to the simulation:
        - Add the new time to the 'Simulation.time' field.
        - Updates the 'TaskStep.current_time' field.
        :param time: The new time to add.
        :return: None
        """
        self.time.append(time)
        self.scheduler.current_time = time
        for task_t in self.task_list.tasks_ts[self.task_list.lsft+1: self.task_list.lst+1]:
            if task_t[0].state in (State.EXECUTING, State.EXECUTING_CALCULATION, State.EXECUTING_IO):
                task_t[0].steps[task_t[0].current_step_index].current_time = time

    def update_energy(self):
        """
        Add a new entry in the list of energy measurements computed by the simulation.
        :return: None
        """
        interval = self.time[-1] - self.time[-2]
        new_total_energy = 0
        for node in self.nodes:
            en = node.power_consumption() * interval
            with open(f"{self.record_folder}/nœuds/{node.name}.csv", "w") as file:
                writer(file).writerow([self.time[-1], en])
            new_total_energy += energy
        for storage_unit in self.storage:
            en = storage_unit.power_consumption() * interval
            with open(f"{self.record_folder}/nœuds/{storage_unit.name}.csv", "w") as file:
                writer(file).writerow([self.time[-1], en, storage_unit.flow, storage_unit.contention])
            new_total_energy += energy
        print("Task statistics to be implemented when File systems are complete")
        for task_t in self.task_list.tasks_ts:
            pass
        self.energy.append(self.energy[-1] + new_total_energy)

    def update_physics(self, time: float):
        """
        Update the fields of time and energy.
        :param time: The new time to introduce.
        :return: None
        """
        assert time >= self.time[-1]
        if self.time[-1] == time:   # If the new time is the same as the previous one, the simulation didn't move.
            # Two transitions will simply happen simultaneously.
            pass
        else:
            self.update_time(time)
            # Update time BEFORE energy.
            self.update_energy()

    def run(self):
        """
        Run the simulation until all tasks are completed.
        :return: Time, energy: the lists of times isolated for the simulation and the energy consumed at these moments.
        """
        if not self.task_list.tasks_ts:
            print(Fore.WHITE + "No task to execute." + Style.RESET_ALL)
            return [0], [0]
        self.event += 1
        self.scheduler.on_new_task(self.task_list.tasks_ts[0][0])
        self.task_list.update_lst()
        while self.next_event[0] != [Event.SIMULATION_TERMINAISON]:
            print(Fore.WHITE + f"{self}" + Style.RESET_ALL)
            if self.next_event[0] == [Event.TASK_SUBMIT]:
                self.scheduler.on_new_task(self.task_list.tasks_ts[self.task_list.lst+1][0])
                self.task_list.update_lst()
            elif self.next_event[0] == [Event.SIMULATION_START]:
                pass
            elif len(self.next_event[0]) == 2:
                # In this case, it means that the last step of a task is finishing, so the whole task must be ended too.
                self.scheduler.on_task_finished(self.next_event[1])
                self.task_list.update_lsft()
            else:
                self.alternate_steps()
            self.update_next_event()
            print(Fore.GREEN + f"\nNEXT_EVENT: {self.next_event}" + Style.RESET_ALL)
            self.update_physics(self.next_event[2])
            self.event += 1
        if len(self.scheduler.queue) > 0:
            print(Fore.BLACK + Back.RED + f"Simulation ended without to execute all tasks. Tasks remaining : "
                                          f"{print_names_from_list(self.scheduler.queue)}" + Style.RESET_ALL)
        return self.time, self.energy


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

    def allocate_space(self, file: "File", disk: Storage):
        # TODO Simplifier la fonction.
        """
        Allocates the required space on a Storage to store a File.
        To succeed, the Storage must have enough free space to allocate the whole file, it can split it if required.
        This implementation searches for the most suitable location for the File in priority: the subspace that is both
        tiniest and of length higher than the File's one.
        If the file is too big to be placed into one unique subspace, it gets split.
        :param file: The incoming File
        :param disk: The receiving Storage.
        :return: None
        """
        # Sort the list of empty spaces with increasing length of the subspaces.
        free_space = sorted(disk.empty_space, key=lambda subspace: subspace[1])
        if free_space[-1][1] >= file.space:   # If there is one subspace large enough to host the whole File:
            ideal_subspace_index = bisect_for_lol(free_space, [0, file.space], 1)
            # To model randomness in space allocation, the beginning byte is 'somewhere' in the subspace.
            interval = free_space[ideal_subspace_index][1] - file.space
            beginning = self.rng.randint(0, interval)
            taken_subspace = [free_space[ideal_subspace_index][0] + beginning, file.space, file]

            # Updating disk's occupied_space.
            noi = bisect_for_lol(disk.occupied_space, taken_subspace, 0)    # noi stands for newly_occupied_index.
            disk.occupied_space.insert(noi, taken_subspace)
            # Check if the modification is correct.
            if disk.occupied_space[noi-1][0] + disk.occupied_space[noi-1][1] > disk.occupied_space[noi][0]:
                print(disk.occupied_space)
                raise ValueError(f"On disk {disk}, allocated subspace {disk.occupied_space[noi-1]} now covers "
                                 f"{disk.occupied_space[noi]}")
            if disk.occupied_space[noi][0] + disk.occupied_space[noi][1] > disk.occupied_space[noi+1][0]:
                print(disk.occupied_space)
                raise ValueError(f"On disk {disk}, allocated subspace {disk.occupied_space[noi - 1]} now covers "
                                 f"{disk.occupied_space[noi]}")

            # Updating disk's free_space. Get back to the location-base sorted list.
            ideal_subspace_index = disk.empty_space.index(free_space[ideal_subspace_index])
            previous_free_space = disk.empty_space[ideal_subspace_index]
            next_occupied_byte = disk.occupied_space[bisect_for_lol(disk.occupied_space, previous_free_space, 0)[0]][0]
            if beginning > 0:   # If the occupied space begins right after the previous one, it is useless to create an
                # "empty" empty space.
                disk.empty_space.insert(ideal_subspace_index, [previous_free_space[0], beginning])
            first_byte_new_free_space = previous_free_space[0] + beginning + file.space  #
            disk.empty_space.insert(ideal_subspace_index+1,
                                    [first_byte_new_free_space,  next_occupied_byte-first_byte_new_free_space])
        else:
            file.split_in_disk = True
            # TODO Coder partie plus dure : scinder le fichier.
            #   On doit pouvoir réutiliser la partie du haut en tant que fonction pour allouer une partie d'un fichier.

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


def original(rng: Random) -> int:
    """
    Returns a digit in base 10 according to a Benford distribution.
    :param rng: A Random instance.
    :return: An int.
    """
    prob = rng.random()  # Pick a position on the [0.0, 1.0) segment, following a uniform distribution.
    k = 1
    cursor = log10(1 + 1/k)  # Cursor that increases on the [0.0, 1.0) axis.
    while cursor <= prob:   # While the cursor hasn't met the position picked up, the number to return increases.
        k += 1
        cursor += log10(1 + 1/k)
    return k


def benford(position: int, rng: Random) -> int:
    """
    Computes a base 10 digit according to Benford law,
    with the approximation of uniform distribution for digits at position superior to the 1st.
    :param position: The position that the returned number will have in the final number.
    :param rng: A Random instance.
    :return: An int.
    """
    if position == 1:
        return original(rng)
    elif position > 1:
        return rng.randrange(10)


def dependencies_generator(tasks: List[Task], rng: Random) -> List[Task]:
    """
    Generates a list of Tasks from
    :param tasks: The list of tasks that have already been created.
    :param rng: A Random instance.
    :return: A list of Tasks.
    """
    if not tasks:
        return []
    deps: List[Task] = []
    deps_count_limit: float = log10(len(tasks))
    deps_count: int = 0
    k: int = 1
    tasks_copy = tasks.copy()
    while deps_count_limit >= k:
        digit: int = benford(k, rng)
        deps_count += int(digit * 10**(floor(deps_count_limit-k)))
        k += 1
    assert deps_count < len(tasks_copy)
    while deps_count >= 0:
        k = rng.randrange(len(tasks_copy))
        deps.append(tasks_copy.pop(k))
        deps_count -= 1
    return deps


def task_generator(list_nodes: List[Node], list_storage: List[Storage],
                   sample_size: int, task_size: int, seed: int)\
        -> TaskSchedulingTrace:
    """
    Generates a TaskSchedulingTrace suitable to run on a Simulation.
    :param list_nodes: List of Nodes that will be used in the simulation to which the set of tasks is created for.
    :param list_storage: List of Storage that will be used in the simulation.
    :param sample_size: The number of tasks to generate
    :param task_size: The average amount of TaskSteps per task.
    :param seed: An integer used to initialise the RNG.
    :return: The set of Tasks and their respective arrival time as a TaskSchedulingTrace instance.
    """
    total_storage_flow: int = 100000000   # The mean of all data to be transferred in an IO step (B).
    flop: int = 100000000000    # The mean of all operations to be performed in a ComputeStep.
    sigma_task = task_size // 10
    max_cores = max(node.core_count for node in list_nodes)    # The value of the maximum number of cores for
    # a single Task.
    io_flow: float = max(store.throughput for store in list_storage)     # The max bandwidth taken on a
    # storage level for an IO step (B/s).
    tasks: List[Task] = []
    timestamps: List[float] = []
    rng = Random()
    rng.seed(seed)
    for k in range(sample_size):
        steps: List[TaskStep] = []
        tasks.append(Task(name=f"task_{k}",
                          min_thread_count=rng.randrange(max_cores),    # We assume that a core takes exactly a thread.
                          dependencies=dependencies_generator(tasks, rng)))
        if timestamps:
            timestamps.append(rng.uniform(timestamps[-1], timestamps[-1] + 5))  # CURRENTLY, the next timestamp is a
            # float taken from a uniform distribution over the interval coming from the last timestamp declared, up
            # to the latter plus 5s.
        else:
            timestamps.append(0)
        number_of_steps = int(rng.gauss(task_size, sigma_task))
        if number_of_steps <= 0:
            number_of_steps = 1
        for kk in range(number_of_steps // 2):  # We create an alternate sequence of IO and Computation steps.
            steps.append(IOTaskStep(task=tasks[-1],
                                    list_storage=[(storage[0], rng.uniform(0., io_flow))],
                                    total_io_volume=rng.randrange(total_storage_flow)))
            steps.append(ComputeTaskStep(task=tasks[-1],
                                         flop=rng.randrange(flop)))
        tasks[-1].add_task_steps(steps)
    return TaskSchedulingTrace(tasks, timestamps)


if __name__ == "__main__":
    # Creating the nodes for our simulation
    nodes = [Node(name=f'cherry{i}', max_frequency=5000000000, min_frequency=1000000000, core_count=core_count)
             for i, core_count in enumerate([64] * 2 + [128] * 4)]
    # print("Nodes: " + ["Empty", "\n  - " + "\n  - ".join(map(str, nodes))][len(nodes) > 0], end="\n\n")

    # Creating the storage for our simulation
    storage = [Storage("HDD", 20000000000000, 500e6, 10e-3)]     ### A Remplacer
    # print(f'Storage: {storage[0]}', end="\n\n")

    # Creating a set of tasks
    # task1 = Task("task1", 28)
    # task2 = Task("task2", 127)
    # cpts1 = ComputeTaskStep(task1, 10000)
    # oits1 = IOTaskStep(task1, [(storage[0], 10000.)], 100000)
    # cpts2 = ComputeTaskStep(task2, 10000)
    # oits2 = IOTaskStep(task2, [(storage[0], 10000.)], 100000)
    # task1.add_task_steps([cpts1, oits1])
    # task2.add_task_steps([cpts2, oits2])

    # Create the trace.
    # Trace = TaskSchedulingTrace([task1, task2], [0, 3])
    trace = task_generator(nodes, storage, 42, 10, 1)

    # Create the scheduler
    scheduler = NaiveScheduler("Naive_Scheduler_v1", nodes, storage)

    # Initialize the simulation
    simulation = Simulation(nodes, storage, trace, scheduler)   ### Add a list of Files at the middle.

    # Run the simulation
    t, energy = simulation.run()

    # Print the results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, energy, 'r-*', label="Énergie consommée")
    ax2.plot(t, gradient(energy, t), 'b-', label="Puissance requise")
    ax2.set_ylim(ymin=0)

    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Énergie (J)', color='r')
    ax2.set_ylabel('Puissance (W)', color='b')

    plt.show()


""" Métriques à ressortir :
Par tâche :
    instant de soumission (connu)
    instant de début d'exécution
    instant de fin d'exécution
    puissance consommée à chaque instant
    énergie consommée au fil du tps
    debit d'I/O au fil du tps
Par nœud :
    Puissance consommée à chaque instant
    énergie dépensée au fil du tps
Par storage :
    idem nœud
Par simulation (on l'exporte déjà en graphique):
    Tps d'exécution de la liste des tâches
    Puissance consommée par le total à chaque instant
    énergie dépensée par le total au fil du tps
"""