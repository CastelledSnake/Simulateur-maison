from typing import List

from tools.tasks.task import Task
from tools.utils import pretty_print


class Node:
    """
    Class describing a compute node.
    """

    def __init__(self, name: str, max_frequency: float, min_frequency: float, core_count: int, static_power: float,
                 sleep_power: float, coefficient_dynamic_power: float, coefficient_leakage_power: float):
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
        self.cdp: float = coefficient_dynamic_power
        self.clp: float = coefficient_leakage_power
        self.busy_cores: int = 0  # Number of cores running a Task on the node.
        self.idle_busy_cores: int = 0  # Number of cores reserved by some Task, but not performing calculations now.
        # By default, we consider that a Task performs calculation on a Node, iff it is in a ComputationTaskStep.
        self.running_tasks: List[Task] = []  # List of Tasks occupying at least one core on the Node.
        self.sleeping: bool = False  # Boolean saying if the Node is switched off ( = True) or not ( = False).

    #   def __setattr__(self, frequency, new_freq: float):
    #       if self.min_frequency <= new_freq <= self.max_frequency:
    #           self.frequency = new_freq

    def __repr__(self):
        return f'Node "{self.name}" running {self.busy_cores}({self.idle_busy_cores})/{self.core_count} ' \
               f'cores for Tasks: {list(map(lambda task: task.name, self.running_tasks))} ' \
               f'with current frequency = {pretty_print(self.frequency, "Hz")}' \
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
        if self.sleeping:
            return self.sleep_power
        else:
            dynamic_power = self.cdp * (self.busy_cores - self.idle_busy_cores) * self.frequency ** 3
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
        self.idle_busy_cores += cores  # Idle cores are (de)allocated by ComputeTaskSteps during the execution.
        if not 0 <= self.idle_busy_cores <= self.busy_cores <= self.core_count:
            raise ValueError(f"{self}")

    def unregister(self, task: Task, cores: int):
        """
        Unregister a Task from all the cores it has on the Node.
        :param task: The task to unregister.
        :param cores: The number of cores the Task has taken on the Node.
        :return: None
        """
        assert 0 <= self.idle_busy_cores <= self.busy_cores <= self.core_count
        self.running_tasks.remove(task)
        self.busy_cores -= cores
        self.idle_busy_cores -= cores
        assert 0 <= self.idle_busy_cores <= self.busy_cores <= self.core_count
