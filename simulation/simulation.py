from _csv import writer
from math import inf
from typing import List, Set
from os import makedirs
from datetime import datetime

from colorama import Style, Fore, Back

from simulation.trace import TaskSchedulingTrace
from simulation.next_event import Event, NextEvent
from simulation.schedule_order import Order, ScheduleOrder
from model.node import Node
from model.storage.abstract_storage_tier import Storage
from model.storage.file import File
from model.storage.hdd_storage_tier import HDD
from model.storage.ssd_storage_tier import SSD
from model.tasks.task import Task, State
from model.utils import pretty_print


def _nature_of_step_end(step_type: State, task: Task, fin_step: float) -> NextEvent:
    """
    Determines what is the exact kind of TaskStep ending (i.e. the type of the step ending and whether the whole)
    Task is finished or not
    :param step_type: The type of the TaskStep ending.
    :param task: The Task to which the Step belongs.
    :param fin_step: The time at which the Step should end.
    :return: A NextEvent object.
    """
    # By default, the step's end is not the task's end, because there may be another step following.
    if step_type == State.EXECUTING_IO:
        next_event = NextEvent({Event.IO_STEP_END}, task, fin_step, None)
    elif step_type == State.EXECUTING_CALCULATION:
        next_event = NextEvent({Event.CALC_STEP_END}, task, fin_step, None)
    else:
        raise ValueError(f"Simulation.step_type has the value '{step_type}', unappropriated for a step_end.")
    if task.current_step_index == len(task.steps) - 1:
        # Here, this is the end of the last step in the task, i.e. the end of the task.
        next_event.events.add(Event.TASK_END)
    return next_event


class Simulation:
    """
    Class describing the simulation instances: each instance of this class is
        a simulation of the execution of a workflow on hardware (both given as arguments).
    """

    def __init__(self, list_nodes: List[Node], list_hdds: List[HDD], list_ssds: List[SSD], list_files: List[File],
                 task_list: TaskSchedulingTrace, algo_scheduler):
        """
        Constructor of the Simulation class
        :param list_nodes: The list of all Nodes in the simulated system.
        :param list_files: The list of all Files initially written on the memory devices.
        :param task_list: The list of all tasks that simulation will have to execute.
        :param algo_scheduler: The scheduling algorithm that will be used in the simulation.
        """
        self.nodes: List[Node] = list_nodes
        self.list_hdds: List[HDD] = list_hdds
        self.list_ssds: List[SSD] = list_ssds
        self.storage: List[Storage] = list_hdds + list_ssds
        self.list_files: List[File] = list_files
        self.task_list: TaskSchedulingTrace = task_list
        self.scheduler = algo_scheduler
        self.next_event: NextEvent = NextEvent(events={Event.SIMULATION_START},
                                               task=self.task_list.tasks_ts[0][0],
                                               time=self.task_list.tasks_ts[0][1],
                                               order=None)
        self.time: List[float] = [0.]  # The different times of interest raised in the simulation
        self.energy: List[float] = [0.]  # Total energy consumption from the beginning of the simulation
        self.event: int = -1  # -1 corresponds to starting of the Simulation.
        self.energies: dict = {}
        for node in self.nodes:
            self.energies[node] = 0.
        for storage in self.storage:
            self.energies[storage] = 0.
        for task_t in self.task_list.tasks_ts:
            self.energies[task_t[0]] = 0.
        self.next_orders: Set[ScheduleOrder] = set()

        # Creation of the folders and files that will contain the results of the Simulation
        now = datetime.now()
        self.record_folder = f"enregistrements_automatiques/{self.scheduler.name}/résultats_du_" \
                             + now.strftime('''%Y-%m-%d_à_%H-%M'-%S"''')
        makedirs(self.record_folder)
        makedirs(f"{self.record_folder}/nœuds")
        makedirs(f"{self.record_folder}/mémoire")
        makedirs(f"{self.record_folder}/tâches")
        makedirs(f"{self.record_folder}/global")
        # Creation of the recording files (one per item of interest).
        with open(f"{self.record_folder}/global/simulation.csv", "x", newline='') as file:
            writer(file).writerow(["Temps (s)", "Énergie (J)"])
            writer(file).writerow([0.0, 0.0])
        for node in self.nodes:
            with open(f"{self.record_folder}/nœuds/{node.name}.csv", "x", newline='') as file:
                writer(file).writerow(["Temps (s)", "Énergie (J)"])
                writer(file).writerow([0.0, 0.0])
        for storage_unit in self.storage:
            with open(f"{self.record_folder}/mémoire/{storage_unit.name}.csv", "x", newline='') as file:
                writer(file).writerow(["Temps (s)", "Énergie (J)", "Débit d'E/S par tâche (O/s)"])
                writer(file).writerow([0.0, 0.0, 0.0])
        for task_t in self.task_list.tasks_ts:
            with open(f"{self.record_folder}/tâches/{task_t[0].name}.csv", "x", newline='') as file:
                writer(file).writerow(["Temps (s)", "Énergie (J)", "Débit d'E/S (O/s)"])
                writer(file).writerow([0.0, 0.0, 0.0])

    def __str__(self):
        r = f'State of the {self.scheduler} at {pretty_print(self.time[-1], "s")}, starting event ' + Fore.RED + \
            f'{self.event}' + Fore.WHITE + f' :\n    Energy consumed : ' \
            f'{pretty_print(self.energy[-1], "J")}\n    Nodes available: '
        r += str(self.nodes)
        r += ';\n    Storage available : '
        r += str(self.storage)
        r += '\n'
        return r

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
            next_event = NextEvent(events={Event.TASK_SUBMIT},
                                   task=self.task_list.tasks_ts[lst+1][0],
                                   time=self.task_list.tasks_ts[lst+1][1],
                                   order=None)
        else:
            next_event = NextEvent({Event.TASK_SUBMIT}, None, inf, None)
        # Searching for the next_event on all launched but non-finished tasks.
        for task_t in self.task_list.tasks_ts[self.task_list.lsft+1: lst+1]:     # task_t stands for "task, time"
            if task_t[0].state in [State.EXECUTING_CALCULATION, State.EXECUTING_IO]:
                fin_step = task_t[0].steps[task_t[0].current_step_index].predict_finish_time(self.time[-1])\
                           + self.time[-1]
                if fin_step <= next_event.time:
                    next_event = _nature_of_step_end(task_t[0].state, task_t[0], fin_step)
        # Searching for the next_event among oncoming ScheduleOrders.
        for order in self.next_orders:
            if order.time <= next_event.time:
                correspondance = [(Order.START_TASK, Event.TASK_BEGIN),
                                  (Order.START_IOTASKSTEP, Event.IO_STEP_BEGIN),
                                  (Order.TRANSFER_FILE, Event.FILE_MOVE_BEGIN)]
                for order_kind, event_kind in correspondance:
                    if order.order == order_kind:
                        next_event = NextEvent({event_kind}, order.task, order.time, order)
        # If no task performs any action, it means all tasks have been executed: this is the end of the simulation.
        if next_event.time == inf:
            next_event = NextEvent({Event.SIMULATION_TERMINAISON}, None, self.time[-1], None)
        self.next_event = next_event

    def alternate_steps(self):
        """
        Performs all te actions required to switch between two TaskSteps.
        :return: None
        """
        task: Task = self.next_event.task
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

    def add_line(self, path: str, element, interval: float):
        """
        Add a line to an output .csv file and returns the energy item written in it.
        :param path: Path to the .csv file
        :param element: The object we want to write metrics for (Storage, Node, Task).
        :param interval: Time interval since the simulation lastly updated its metrics.
        :return: The energy consumed by 'element' in 'interval'. float
        """
        energy = element.power_consumption() * interval
        with open(path, "a", newline='') as file:
            if type(element) == Node:
                writer(file).writerow([self.time[-1], self.energies[element] + energy])
            elif type(element) == HDD or SSD:
                writer(file).writerow([self.time[-1], self.energies[element] + energy,
                                       element.get_current_throughput_per_task('r')])
            else:
                raise TypeError(f"'{element} has type {type(element)}, not implemented.")
        self.energies[element] += energy
        return energy

    def update_energy(self):
        """
        Add a new entry in the list of energy measurements computed by the simulation.
        :return: None
        """
        interval = self.time[-1] - self.time[-2]
        new_total_energy = 0
        for node in self.nodes:
            energy = self.add_line(f"{self.record_folder}/nœuds/{node.name}.csv", node, interval)
            new_total_energy += energy
        for storage_unit in self.storage:
            energy = self.add_line(f"{self.record_folder}/mémoire/{storage_unit.name}.csv", storage_unit, interval)
            new_total_energy += energy
        # TODO Task statistics to be implemented.
        for task_t in self.task_list.tasks_ts:
            # Définition de l'énergie prise par une tâche :
            # Pour les nœuds :
            #   C'est l'énergie consommée par l'ensemble des cœurs attribués à la tâche (qu'ils calculent où soient au
            #   repos) PLUS l'énergie 'perdue' pour le fonctionnement du nœud lui-même divisée par le nombre de tâches
            #   inscrites sur le nœud PLUS l'énergie consommée par les éventuels nœuds en veille divisée par le nombre
            #   total de tâches en cours d'exécution.
            # Pour les disques :
            #   C'est l'énergie consommée par le disque contenant le fichier auquel accède la tâche divisée par le
            #   nombre de tâches accédant à un fichier sur ce même disque (on rappelle que la BP est répartie
            #   équitablement) PLUS l'énergie consommée par les éventuels disques en veille divisée par le nombre total
            #   de tâches en cours d'exécution.
            pass
        with open(f"{self.record_folder}/global/simulation.csv", "a", newline='') as file:
            writer(file).writerow([self.time[-1], self.energy[-1]+new_total_energy])
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
        while self.next_event.events != {Event.SIMULATION_TERMINAISON}:
            print(Fore.WHITE + f"{self}" + Style.RESET_ALL)
            new_orders: List[ScheduleOrder] = []
            if self.next_event.events == {Event.SIMULATION_START}:
                # This event leads to TASK_SUBMIT(task_0).
                assert self.event == -1
            elif self.next_event.events == {Event.TASK_SUBMIT}:
                self.task_list.tasks_ts[self.task_list.lst+1][0].state = State.SUBMITTED
                new_orders = self.scheduler.on_new_task(self.task_list.tasks_ts[self.task_list.lst+1][0])
                self.task_list.update_lst()
            elif Event.TASK_END in self.next_event.events:
                new_orders = self.scheduler.on_task_finished(self.next_event.task)
                self.task_list.update_lsft()
            elif next(iter(self.next_event.events)) in {Event.CALC_STEP_BEGIN, Event.IO_STEP_BEGIN,
                                                        Event.CALC_STEP_END, Event.IO_STEP_END}:
                self.alternate_steps()
            elif self.next_event.events == {Event.TASK_BEGIN}:
                assert self.next_event.order.time == self.next_event.time == self.time[-1]
                self.next_event.order.task.on_start(self.next_event.order.nodes, self.next_event.order.time)
                self.scheduler.task_executed(self.next_event.order)
                self.next_orders.remove(self.next_event.order)
            elif self.next_event.events == {Event.FILE_MOVE_BEGIN}:
                raise NotImplementedError
            else:
                raise NotImplementedError(f"next event has type {self.next_event.events}, not implemented")
            self.next_orders.update(new_orders)
            self.update_next_event()
            print(Fore.GREEN + repr(self.next_event) + Style.RESET_ALL)
            self.update_physics(self.next_event.time)
            self.event += 1
        if self.scheduler.queue:
            print(Fore.BLACK + Back.RED + f"Simulation ended without to execute all tasks. Tasks remaining : "
                                          f"{list(map(str, self.scheduler.queue))}" + Style.RESET_ALL)
        remaining: List[Task] = []
        for task_t in self.task_list.tasks_ts:
            if task_t[0].state != State.FINISHED:
                remaining.append(task_t[0])
        if remaining:
            print(Fore.BLACK + Back.RED + f"Simulation ended without to finish all tasks.  Tasks not ended : "
                                      f"{list(map(str, remaining))}" + Style.RESET_ALL)
        return self.time, self.energy
