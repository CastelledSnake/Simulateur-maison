# PREMIERE SECTION
# from collections import deque, defaultdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from random import Random
from enum import Enum, auto
from matplotlib import pyplot as plt
from numpy import gradient, inf, log10, floor
# from bisect import insort
from colorama import Back, Fore, Style
# FIN PREMIERE SECTION


# DEUXIÈME SECTION
def pp_unit(value: int or float, unit: str, round_size: int = 3):
    # TODO : recoder le simulateur avec des entiers uniquement, de précision jusqu'à 10**-30 par rapport aux unités.
    """
    Pretty-prints an integer with adequate unit, multiple and size.
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
    elif value >= 1e-15:     # femto
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
    :param liste: A list of elements such as each of these elements have a field "name".
    :return: A string nicely printing each element's name.
    """
    string: str = "["
    for k in range(len(liste)):
        string += liste[k].name
        if k < len(liste) - 1:
            string += ", "
    string += "]"
    return string


def print_hardware_list(liste: List) -> str:
    """
    Nicely print a list of hardware devices (Node and Storage).
    :param liste: A list of elements to print.
    :return: a string.
    """
    string = "["
    for k in range(len(liste)):
        string += f'{liste[k]}'
        if k < len(liste) - 1:
            string += ', '
    string += ']'
    return string


class State(Enum):
    """
    Describes the possible states for tasks, from their creation (out of simulation range) to their terminaison.
    """

    NOT_SUBMITTED = auto()  # The task has just been defined : the simulated system haven't heard of it yet.
    QUEUED = auto()  # The task has been submitted to the scheduler, but no resource is allocated for it.
    EXECUTING = auto()  # The task has been allocated computation resources, but isn't in a calculation or I/O phase
    EXECUTING_CALCULATION = auto()  # The task is performing some calculations.
    EXECUTING_IO = auto()  # The task is performing some I/O.
    FINISHED = auto()  # The task is completed : terminated.


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
# FIN DEUXIÈME SECTION


# TROISIÈME SECTION
class TaskStep(ABC):
    """
    Mother class of ComputeTaskStep and IOTaskStep.
    """

    def __init__(self, task: "Task"):
        """
        Constructor of TaskStep class.
        :param task: The task to which belongs the step.
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
            considering resources allocated and assuming there is no perturbations incoming in the system.
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
        :param task: The task to which belongs the compute step.
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
            considering resources allocated and assuming there is no perturbations incoming in the system.
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
    Describes a step of a task that is dedicated to I/O.
    """

    def __init__(self, task: "Task", list_storage: List[Tuple["Storage", float]], total_io_volume: int):
        """
        Constructor of IOTaskStep class.
        :param task: The task to which belongs the IO step.
        :param list_storage: The list of storage instances that are concerned with IO of this step,
            and the bandwidth (B/s) the task requires on each one.
            The occupation is considered as constant for the whole execution.
        :param total_io_volume: The total amount of IO required to complete the step (B)
        """
        TaskStep.__init__(self, task)
        self.list_storage: List[Tuple["Storage", float]] = list_storage
        # TODO : Qui choisit la position des données au sein des différentes ressources de stockage ?
        #   (la simu, le scheduler, la mémoire elle-même, la tâche, l'étape de tâche ?)
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
        self.list_storage[0][0].register(self.task, self.list_storage[0][1])

    def on_finish(self):
        """
        End the IOTaskStep, liberate storage resources.
        :return: None
        """
        assert self.task.state is State.EXECUTING_IO
        self.list_storage[0][0].unregister(self.task, self.list_storage[0][1])
        self.task.state = State.EXECUTING

    def finish_time(self):
        """
        Computes an estimation of the remaining time to complete the IOTaskStep,
            considering resources allocated and assuming there is no perturbations incoming in the system.
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
        # Debug version of previous __str__ : more detailed, but more verbose.
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
        All the parameters when calling this method must have been decided by the Scheduler.
        :param list_nodes: A list of tuples, each containing one node on which task shall execute,
        and the amount of cores to reserve on this node.
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
                 static_power: float = 100., sleep_power: float = 0., coefficient_for_dynamic_power: float = 1e-28):
        """
        Constructor of Node class.
        :param name: The name of the Node (str).
        :param max_frequency: The maximum frequency that Node's clock can reach (Hz).
        :param min_frequency: The minimum frequency that Node's clock can have (Hz).
        :param core_count: Total number of cores on the Node (int).
        :param static_power: Basis power consumption (W) of the node, it's the minimal power consumption of the Node
        when it's not sleeping.
        :param sleep_power: Power consumption (W) of the node when it's switched off.
        """
        self.name: str = name
        self.max_frequency: float = max_frequency
        self.min_frequency: float = min_frequency
        self.frequency: float = max_frequency  # By default, the Node takes the highest frequency it can.
        self.core_count: int = core_count
        self.static_power: float = static_power
        self.sleep_power: float = sleep_power
        self.k = coefficient_for_dynamic_power  # A multiplicative coefficient that is inserted in the equation of
        # power consumption. In the original study describing the model, it varies from 0.4 to 17.50, depending on the
        # device. The study : https://www.sciencedirect.com/science/article/abs/pii/S2210537916301755
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
               f'with current frequency = {pp_unit(self.frequency, "Hz")}' \
            # f"taking powers: [max = {pp_unit(self.busy_core_power, 'W')}, " \
        # f"min = {pp_unit(self.idle_core_power, 'W')}, idle = {pp_unit(self.sleep_power, 'W')}] "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this Node.
        :return: The power consumption (W) as a float.
        This formula comes from this study : https://www.sciencedirect.com/science/article/abs/pii/S2210537916301755
        """
        return self.static_power + (self.k * (self.busy_cores-self.idle_busy_cores) * self.frequency**3)

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


class File:
    """
    Class describing a file on the system. These files can be red or written by IOTaskSteps, allowing to model I/Os.
    """
    def __init__(self):
        raise NotImplementedError


class Storage:
    """
    Class describing a storage device (HDD, SSD, RAM, ...).
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
    # On introduit un débit max en sortie de chaque noeud.
    def __init__(self, name: str, capacity: int, throughput: float, latency: float,
                 files: List[File], power: float = 0.):
        """
        Constructor of the Storage class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required to communicate any piece of data with any other device of the simulation.
        :param power: Power (W) required to run
                        0 by default, because storage consumption never appears in papers about I/O.
        """
        self.name: str = name
        self.capacity: int = capacity
        self.throughput: float = throughput
        self.latency: float = latency
        self.files: List[File] = files
        self.power: float = power
        self.occupation: float = 0.  # Raw amount of I/O-bandwidth (B/s) used at this time.
        self.running_tasks: List[Task] = []
        self.contention: float = 1.  # A number useful to compute delays linked with I/O-contention
        #   si occupation <= throughput : 1 (aucun problème)
        #   si occupation > throughput : occupation / throughput
        #       (le temps mis pour compléter une quantité d'I/O devient temps_initial * contention,
        #       et la BP réelle disponible pour une tâche devient contention / bp_initiale).

    def __str__(self):
        if self.power:
            return f'Storage "{self.name}" with capacity = {pp_unit(self.capacity, "B")} ' \
                   f'throughput = {pp_unit(self.occupation, "B/s")} in {pp_unit(self.throughput, "B/s")} ' \
                   f'for Tasks: {print_names_from_list(self.running_tasks)} ' \
                   f'contention = {self.contention} ' \
                   f'taking {pp_unit(self.power, "W")}'
        # f"latency = {pp_unit(self.latency, 's')} " \
        return f'Storage "{self.name}" with capacity = {pp_unit(self.capacity, "B")} ' \
               f'throughput = {pp_unit(self.occupation, "B/s")} in {pp_unit(self.throughput, "B/s")} ' \
               f'contention = {self.contention}'
        # f"latency = {pp_unit(self.latency, 's')} " \

    def update_contention(self):
        """
        Determine if the storage device experiences I/O-contention
            and modifies the contention instance variable consequently.
        :return: None
        """
        if self.occupation <= self.throughput:
            self.contention = 1.
        else:
            self.contention = self.occupation / self.throughput

    def register(self, task: Task, bandwidth: float):
        """
        Give a Task part of the available  I/O bandwidth.
        :param task: Newly elected Task for an I/O booking.
        :param bandwidth: The amount of bandwidth to allocate to the Task.
        :return: None
        """
        assert task.steps[task.current_step_index].step_type == TaskStepType.IO
        assert self.occupation >= 0
        assert bandwidth >= 0
        self.occupation += bandwidth    # Currently, the task_step decide which amount of bp to be taken on each Storage
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
        delta = self.occupation - bandwidth
        if delta < 0:
            if abs(delta) > 1e-6:   # If the delta is below 0, and lower than 1 µs, the error is probably due to
                # floating point imprecision.
                raise ValueError(f"The occupation {self.occupation} of Storage {self.name} is about to be lowered of "
                                 f"{bandwidth}")
            else:
                self.occupation = 0.
        else:
            self.occupation -= bandwidth
        self.running_tasks.remove(task)
        self.update_contention()


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
        :param task: the oncoming task
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
    such as each task is associated with a time of submission (i.e. it becomes candidate to be executed).

    ATTENTION : this class is just a way to store data.
    """

    def __init__(self, tasks: List[Task], task_submission_timestamps: List[float]):
        """
        Constructor of TaskSchedulingTrace class.
        :param tasks: The list of tasks that simulation shall execute.
        :param task_submission_timestamps: The list of moments at which each task becomes candidate for execution.
        """
        self.tasks_ts: List[Tuple[Task, float]] = sorted([(task, timestamp) for task, timestamp
                                                          in zip(tasks, task_submission_timestamps)],
                                                         key=lambda t_ple: t_ple[1])
        self.lsft: int = -1    # The index of the "Last Surely Finished Task" of the simulation
        # Be careful, it is not the index of the lastly finished Task,
        # it is such as all tasks before it, in the order of their submission timestamps, are already finished.
        self.lst: int = -1    # The index of the "Last Submitted Task" to simulator's scope.
        # Note that between the indexes of lft and lst, there can be no Task already submitted,
        # because the Tasks are ordered according to their submission timestamps.
        # But there can be some already finished Tasks, because their purpose is only to focus the simulation on
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
    Class describing the simulation instances : each instance of this class is
        a simulation of the execution of a workflow on a hardware (both given as arguments).
    """

    def __init__(self, list_nodes: List[Node], list_storage: List[Storage],
                 task_list: TaskSchedulingTrace, algo_scheduler):
        """
        Constructor of the Simulation class
        :param list_nodes: The list of all available computation nodes in the simulated system.
        :param list_storage: The list of all storage devices in the simulated system.
        :param task_list: The list of all tasks that simulation will have to execute.
        :param algo_scheduler: The scheduling algorithm that will be used in the simulation.
        """
        self.nodes: List[Node] = list_nodes
        self.storage: List[Storage] = list_storage
        self.task_list: TaskSchedulingTrace = task_list
        self.scheduler = algo_scheduler
        self.next_event: List[Union[List[Event], Task, float]] =\
            [[Event.SIMULATION_START], self.task_list.tasks_ts[0][0], self.task_list.tasks_ts[0][1]]
        self.time: List[float] = [0.]  # The different times of interest raised in the simulation
        self.energy: List[float] = [0.]  # Total energy consumption from the beginning of the simulation
        self.event: int = -1

    def __str__(self):
        r = f'State of the {self.scheduler} at {self.time[-1]}s, event {self.event} :\nEnergy consumed : ' \
            f'{pp_unit(self.energy[-1], "J")}\nNodes available: '
        r += print_hardware_list(self.nodes)
        r += '; Storage available : '
        r += print_hardware_list(self.storage)
        r += '\n'
        return r

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
            # Here, this is the end of the last step of the task, i.e. the end of the task.
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
        # Initialisation of variable next_event.
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
        # If no task perform any action, it means all tasks have been executed : this is the end of the simulation.
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
        new_energy = 0
        for node in self.nodes:
            if node.sleeping:
                new_energy += interval * node.sleep_power
            else:
                new_energy += interval * node.power_consumption()
        for storage_unit in self.storage:
            new_energy += interval * storage_unit.power
        self.energy.append(self.energy[-1] + new_energy)

    def update_physics(self, time: float):
        """
        Update the fields of time and energy.
        :param time: the new time to introduce.
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
        :return: time, energy: the lists of times isolated for the simulation and the energy consumed at these moments.
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

    def __init__(self, env_nodes: List[Node], env_storage: List[Storage]):
        """
        Constructor of the NaiveScheduler class.
        :param env_nodes: List of nodes on the system.
        :param env_storage: List of storage devices on the system.
        """
        # List containing all the available resources :
        self.nodes: List[Node] = env_nodes
        self.storage: List[Storage] = env_storage
        # Queue of all candidate tasks that cannot be executed yet, because of a lack of available resources :
        self.queue: List[Task] = []
        self.current_time: float = 0.
        # TODO On considère que c'est à l'ordonnanceur de gérer les dépendances. Est-ce bien ce qu'on veut ?
        self.finished_tasks: List[Task] = []    # The list of all tasks that are already finished.

    def queuing(self, task: Task, insertion=-1) -> None:
        """
        Performs the required operations to put an oncoming task at the end of the queue.
        :param task: The task to put in the queue.
        :param insertion: an int saying, in case the execution of the Task fails, where to insert it in the queue.
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
        :return: A list containing:
            a list of tuples in which the 1st element is a node and the 2nd is its number of available cores.
            the total of available cores in the system.
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
        Check if all Tasks for which completion was required by another one, have eventually been satisfied or not.
        :param task: The Task that requires the completion of others.
        :return: True if all dependencies are done, False otherwise.
        """
        deps: List[Task] = task.dependencies
        k: int = 0
        exe: bool = len(deps) <= len(self.finished_tasks)     # If the Task requires more Tasks to be finished than
        # the ones that eventually are, the case is closed.
        while exe and k < len(deps):
            if deps[k] not in self.finished_tasks:
                exe = False
            k += 1
        return exe

    def executing_task(self, task: Task, insertion: int = 0) -> List[Tuple[Node, int]] or None:
        """
        Check if there is enough resources to execute the task in argument.
        - If yes, try to allocate all the resources needed for it and returns the list of nodes & cores to allocate.
        - If no, put the task at the end of the queue and return None.
        :param task: The task to execute
        :param insertion: an int saying, in case the execution of the Task fails, where to insert it in the queue.
        :return: The list of couples (node, cores) to allocate to the task, or None if the task cannot be executed.
        """
        m_cores = task.min_thread_count  # We assume that one thread occupies exactly one core.
        [av_nodes, tot_av_cores] = self.sort_least_used_nodes()
        if tot_av_cores < m_cores:
            # If there is not enough calcul capacities available, we don't try to execute the application.
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
    :return: an int.
    """
    prob = rng.random()  # Pick a position on the [0.0, 1.0) segment, following a uniform distribution.
    k = 1
    cursor = log10(1 + 1/k)  # Cursor that increases on the [0.0, 1.0) axis.
    while cursor <= prob:   # While the cursor haven't met the position picked up, the number to return increases.
        k += 1
        cursor += log10(1 + 1/k)
    return k


def benford(position: int, rng: Random) -> int:
    """
    Computes a base 10 digit according to Benford law,
    with the approximation of uniform distribution for digits at position superior to the 1st.
    :param position: The position that the returned number will have in the final number.
    :param rng: A Random instance.
    :return: an int.
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
    :return: a list of Tasks.
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
    :param sample_size: The amount of tasks to generate
    :param task_size: The average amount of TaskSteps per task.
    :param seed: An integer used to initialise the RNG.
    :return: The set of Task and their respective arrival time as a TaskSchedulingTrace instance.
    """
    total_storage_occupation: int = 100000000   # The mean of the amount of data to transfer for an IO step (B).
    flop: int = 100000000000    # The mean of the amount of operations to perform for a compute step.
    sigma_task = task_size // 10
    max_cores = max(node.core_count for node in list_nodes)    # The value of the maximum number of cores for
    # a single Task.
    bandwidth_occupation: float = max(store.throughput for store in list_storage)     # The max bandwidth taken on a
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
                                    list_storage=[(storage[0], rng.uniform(0., bandwidth_occupation))],
                                    total_io_volume=rng.randrange(total_storage_occupation)))
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
    # In our first iteration, there is only a single storage tier
    storage = [Storage("HDD", 20000000000000, 500e6, 10e-3, [], 0)]
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
    # trace = TaskSchedulingTrace([task1, task2], [0, 3])
    trace = task_generator(nodes, storage, 42, 10, 1)

    # Create the scheduler
    scheduler = NaiveScheduler(nodes, storage)

    # Initialize the simulation
    simulation = Simulation(nodes, storage, trace, scheduler)

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
