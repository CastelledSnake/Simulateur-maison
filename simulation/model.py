from _csv import writer
from math import inf
from typing import List, Set
from os import makedirs

from colorama import Style, Fore, Back

from simulation.trace import TaskSchedulingTrace
from simulation.next_event import Event, NextEvent
from simulation.schedule_order import Order, ScheduleOrder
from tools.node import Node
from tools.storage.abstract_storage_tier import Storage
from tools.storage.file import File
from tools.storage.hdd_storage_tier import HDD
from tools.storage.ssd_storage_tier import SSD
from tools.tasks.task import Task, State
from tools.utils import pretty_print


def _nature_of_step_end(step_type: State, task: Task, fin_step: float) -> NextEvent:
    """
    Determines what is the exact kind of TaskStep ending (i.e. the type of the step ending and whether the whole)
    Also states if Task is finished or not.
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


class Model:
    """
    Class describing the simulation instances: each instance of this class is
        a simulation of the execution of a workflow on hardware (both given as arguments).
    """

    def __init__(self, nodes: List[Node], hdds: List[HDD], ssds: List[SSD], list_files: List[File],
                 tasks_trace: TaskSchedulingTrace):
        """
        Constructor of the Simulation class
        :param nodes: The list of all Nodes in the simulated system.
        :param list_files: The list of all Files initially written on the memory devices.
        :param tasks_trace: The list of all tasks that simulation will have to execute.
        """
        self.nodes: List[Node] = nodes
        self.hdds: List[HDD] = hdds
        self.sdds: List[SSD] = ssds
        self.storage: List[Storage] = hdds + ssds
        self.files: List[File] = list_files
        self.tasks_trace: TaskSchedulingTrace = tasks_trace
        self.next_event: NextEvent or None = None
        self.time: List[float] = [0.]  # The different times of interest raised in the simulation
        self.energy: List[float] = [0.]  # Total energy consumption from the beginning of the simulation
        self.event: int = -1  # -1 corresponds to starting of the Simulation.
        self.energies: dict = {}
        for node in self.nodes:
            self.energies[node] = 0.
        for storage in self.storage:
            self.energies[storage] = 0.
        for task_t in self.tasks_trace.tasks_ts:
            self.energies[task_t[0]] = 0.
        self.next_orders: Set[ScheduleOrder] = set()
        self.record_folder: str = ""

    def __str__(self, scheduler=None):
        if scheduler:
            r = f'State of the model(' + Fore.RED + f'{scheduler.name}' + Fore.WHITE \
                + f') at {pretty_print(self.time[-1], "s")}, starting event ' + Fore.RED + f'{self.event}' \
                + Fore.WHITE + f' :\n    Energy consumed : ' \
                               f'{pretty_print(self.energy[-1], "J")}\n    Nodes available: '
        else:
            r = f'State of the ' + Fore.LIGHTRED_EX + f'light ' + Fore.WHITE \
                + f'model at {pretty_print(self.time[-1], "s")}, starting event ' + Fore.RED + f'{self.event}' \
                + Fore.WHITE + f' :\n    Energy consumed : ' \
                               f'{pretty_print(self.energy[-1], "J")}\n    Nodes available: '
        r += str(self.nodes)
        r += ';\n    Storage available : '
        r += str(self.storage)
        r += '\n'
        return r

    def initiate_folders(self, record_folder: str):
        """
        Creates the folders and files that will contain the results of the Simulation
        """
        self.record_folder: str = record_folder
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
        for task_t in self.tasks_trace.tasks_ts:
            with open(f"{self.record_folder}/tâches/{task_t[0].name}.csv", "x", newline='') as file:
                writer(file).writerow(["Temps (s)", "Énergie (J)", "Débit d'E/S (O/s)"])
                writer(file).writerow([0.0, 0.0, 0.0])

    def add_line(self, path: str, element, value: float or List[float]):
        """
        Add a line to an output .csv file, containing given value(s).
        :param path: Path to the .csv file
        :param element: The object we want to write metrics for (Storage, Node, Task).
        :param value:
        :return: The energy consumed by 'element' in 'interval'. float
        """
        with open(path, "a", newline='') as file:
            if type(element) == Node:
                assert type(value) == float
                writer(file).writerow([self.time[-1], value])
            elif type(element) == HDD or SSD:
                assert type(value) == list
                writer(file).writerow([self.time[-1], value[0], value[1]])
            elif type(element) == Task:
                pass
            else:
                raise TypeError(f"'{element} has type {type(element)}, not implemented.")

    def update_next_event(self, scheduler):
        """
        Determines what is the next event in the simulation.
        It can be a task completion, a task arrival or the end of the simulation.
        :param scheduler:
        :return: None
        """
        print(Fore.BLUE + f"lsti : {self.tasks_trace.lsti}, lsfti : {self.tasks_trace.lsfti}" + Style.RESET_ALL)
        lsti = self.tasks_trace.lsti
        assert type(lsti) is int
        # Initialization of variable next_event.
        if (lsti < len(self.tasks_trace.tasks_ts) - 1) and scheduler:
            next_event = NextEvent(events={Event.TASK_SUBMIT},
                                   task=self.tasks_trace.tasks_ts[lsti + 1][0],
                                   time=self.tasks_trace.tasks_ts[lsti + 1][1],
                                   order=None)
        else:
            next_event = NextEvent({Event.TASK_SUBMIT}, None, inf, None)
        # Searching for the next_event among oncoming ScheduleOrders.
        for order in self.next_orders:
            if order.time <= next_event.time:
                correspondance = [(Order.START_TASK, Event.TASK_BEGIN),
                                  (Order.START_IOTASKSTEP, Event.IO_STEP_BEGIN),
                                  (Order.TRANSFER_FILE, Event.FILE_MOVE_BEGIN)]
                for order_kind, event_kind in correspondance:
                    if order.order == order_kind:
                        next_event = NextEvent({event_kind}, order.task, order.time, order)
        # Searching for the next_event on all launched but non-finished tasks.
        for task_t in self.tasks_trace.tasks_ts[self.tasks_trace.lsfti + 1: lsti + 1]:  # task_t stands for "task, time"
            if task_t[0].state in [State.EXECUTING_CALCULATION, State.EXECUTING_IO]:
                fin_step = task_t[0].steps[task_t[0].current_step_index].predict_finish_time(self.time[-1]) \
                           + self.time[-1]
                if fin_step <= next_event.time:
                    next_event = _nature_of_step_end(task_t[0].state, task_t[0], fin_step)
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
        for task_t in self.tasks_trace.tasks_ts[self.tasks_trace.lsfti + 1: self.tasks_trace.lsti + 1]:
            if task_t[0].state in (State.EXECUTING, State.EXECUTING_CALCULATION, State.EXECUTING_IO):
                task_t[0].steps[task_t[0].current_step_index].current_time = time

    def update_energy(self):
        """
        Add a new entry in the list of energy measurements computed by the simulation.
        :return: None
        """
        interval = self.time[-1] - self.time[-2]
        new_total_energy = 0
        output = []
        for node in self.nodes:
            energy = node.power_consumption() * interval
            self.energies[node] += energy
            new_total_energy += energy
            output.append(self.energies[node])
        for storage_unit in self.storage:
            energy = storage_unit.power_consumption() * interval
            self.energies[storage_unit] += energy
            new_total_energy += energy
            throughput_per_task = storage_unit.get_current_throughput_per_task('r')
            output.append([self.energies[storage_unit], throughput_per_task])
        # TODO Task statistics to be implemented.
        for task_t in self.tasks_trace.tasks_ts:
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
        self.energy.append(self.energy[-1] + new_total_energy)
        return output

    def update_physics(self, time: float) -> List:
        """
        Update the fields of time and energy.
        :param time: The new time to introduce.
        :return: None
        """
        assert time >= self.time[-1] or self.time[-1] - time < 1e-4
        if self.time[-1] == time:  # If the new time is the same as the previous one, the simulation didn't move.
            # Two transitions will simply happen simultaneously.
            return []
        else:
            self.update_time(time)
            # Update time BEFORE energy.
            return self.update_energy()

    def save_physics(self, output: List):
        """
        Saves the values in 'output' into the appropriated files.
        Is intended to use the output of Scheduler.update_energy()
        :param output: A list of numbers to be written in record logs. The scheme is to be the one of
        Scheduler.update_energy() output.
        :return: None
        """
        assert self.record_folder
        if output:
            for node in self.nodes:
                value: float = output.pop(0)
                self.add_line(f"{self.record_folder}/nœuds/{node.name}.csv", node, value)
            for storage_unit in self.storage:
                value: List[float] = output.pop(0)
                self.add_line(f"{self.record_folder}/mémoire/{storage_unit.name}.csv", storage_unit, value)
            with open(f"{self.record_folder}/global/simulation.csv", "a", newline='') as file:
                writer(file).writerow([self.time[-1], self.energy[-1]])

    def simulation_start(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute the simulation start.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        # This event only leads to TASK_SUBMIT(task_0).
        if scheduler:  # If a scheduler is given, simulation start must happen only as the very first event.
            assert self.event == -1
        return new_orders

    def task_submit(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute a Task submitting.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        # If no scheduler is given, and instruction pointer is here, it means that this simulation tries to submit
        # a Task while we don't want it to, because no scheduler implies it is a light_model, so all not_submitted
        # Tasks have been removed.
        if not scheduler:
            raise AssertionError(f"This simulation tries to submit a new Task to a non existing Scheduler.")
        self.tasks_trace.tasks_ts[self.tasks_trace.lsti + 1][0].state = State.SUBMITTED
        self.tasks_trace.update_lsti()  # Updates the interesting window of the tasks_trace.
        new_orders = scheduler.on_new_task(self.tasks_trace.tasks_ts[self.tasks_trace.lsti][0])  # Triggers the
        # Scheduler to get some orders.
        return new_orders

    def task_begin(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute a Task begin.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        assert self.next_event.order.time == self.next_event.time == self.time[-1]  # Assert that the Task is launched
        # at the planned moment.
        self.next_event.task.on_start(self.next_event.order.nodes, self.next_event.order.time)  # Begins the Task.
        if scheduler:
            scheduler.task_launched(self.next_event.order)  # Informs the Scheduler that the Task is now executing.
        self.next_orders.remove(self.next_event.order)  # Remove the order from the remaining ones.
        return new_orders

    def task_end(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute a Task ending.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        # If a scheduler is given, the simulation must inform it that a Task has just finished, to get some orders.
        # Otherwise, the simulation must already have a full agenda for all Tasks in its scope. So no need for an order.
        self.next_event.task.on_finish()
        if scheduler:
            print(Fore.RED + repr(self.next_event.task) + Style.RESET_ALL)
            new_orders = scheduler.on_task_finished(self.next_event.task)
        self.tasks_trace.update_lsfti()  # Updates the interesting window of the tasks_trace.
        return new_orders

    def new_step(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute a new TaskStep.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        # For now, we only use STEP_END, no STEP_BEGIN.
        self.alternate_steps()
        return new_orders

    def file_move_begin(self, new_orders: List[ScheduleOrder], scheduler=None):
        """
        Performs the required computations to execute a File move between 2 disks.
        :param new_orders: List of Schedule_Orders to return, if any.
        :param scheduler: The Scheduler that the simulation uses, if any.
        :return:
        """
        raise NotImplementedError

    def simulation_step(self, record_folder: str = "", scheduler=None):
        """
        Performs a simulation step : an evolution of the Model from one state to another.
        :param record_folder:
        :param scheduler:
        :return:
        """
        # Displays the current system state in the console.
        print(Fore.GREEN + repr(self.next_event) + Style.RESET_ALL)
        print(Fore.WHITE + self.__str__(scheduler) + Style.RESET_ALL)
        new_orders: List[ScheduleOrder] = []

        # Consider what to do, according to the content of self.next_event
        if self.next_event.events == {Event.SIMULATION_START}:
            new_orders = self.simulation_start(new_orders, scheduler)
        elif self.next_event.events == {Event.TASK_SUBMIT}:
            new_orders = self.task_submit(new_orders, scheduler)
        elif self.next_event.events == {Event.TASK_BEGIN}:
            new_orders = self.task_begin(new_orders, scheduler)
        elif Event.TASK_END in self.next_event.events:
            new_orders = self.task_end(new_orders, scheduler)
        elif next(iter(self.next_event.events)) in {Event.CALC_STEP_BEGIN, Event.IO_STEP_BEGIN,
                                                    Event.CALC_STEP_END, Event.IO_STEP_END}:
            new_orders = self.new_step(new_orders, scheduler)
        elif self.next_event.events == {Event.FILE_MOVE_BEGIN}:
            new_orders = self.file_move_begin(new_orders, scheduler)
        elif self.next_event.events == {Event.SIMULATION_TERMINAISON}:
            pass
        else:
            raise NotImplementedError(f"next event has type {self.next_event.events}, not implemented")

        # Update metrics and determines the next event.
        self.next_orders.update(new_orders)
        self.update_next_event(scheduler)
        output = self.update_physics(self.next_event.time)
        print(f"RECORD FOLDER : {record_folder}")
        if record_folder:
            self.save_physics(output)
        self.event += 1

    def simulate(self, record_folder: str = "", scheduler=None):
        """
        Run the simulation until all tasks are completed.
        :param record_folder: the record folder for the simulation's metrics. If None, no save is performed.
        :param scheduler: The Scheduler that this simulation will use. If None, the model is 'light', which means
        this simulation is called by a Scheduler in order to get data on what's next.
        :return: Time, energy: the lists of times isolated for the simulation and the energy consumed at these moments.
        """
        # Initialising variables, and asserting there are some Tasks to execute.
        # TODO : Quand une light_sim se déclenche avec simulate, la ligne suivante la démarre à t=0, pas très cohérent,
        #  mais pas bloquant.
        self.next_event: NextEvent = NextEvent(events={Event.SIMULATION_START},
                                               task=None,
                                               time=0.,
                                               order=None)
        if record_folder:
            self.initiate_folders(record_folder)
        if not self.tasks_trace.tasks_ts:
            print(Fore.GREEN + repr(self.next_event) + Style.RESET_ALL)
            print(Fore.WHITE + "No task to execute." + Style.RESET_ALL)
            return [0], [0]

        # Simulation loop : while there are activities to perform, simulation walks to its next event.
        while self.next_event.events != {Event.SIMULATION_TERMINAISON}:
            self.simulation_step(record_folder, scheduler)

        # Some assertion to be sure all model's Tasks have been launched and finished.
        if type(scheduler).__name__ == "NaiveScheduler" and scheduler.queue:
            raise AssertionError(Fore.BLACK + Back.RED
                                 + f"Simulation ended without to execute all tasks. Tasks remaining : "
                                   f"{list(map(str, scheduler.queue))}" + Style.RESET_ALL)
        remaining: List[Task] = []
        for task_t in self.tasks_trace.tasks_ts:
            if task_t[0].state != State.FINISHED:
                remaining.append(task_t[0])
        if remaining:
            raise AssertionError(Fore.BLACK + Back.RED
                                 + f"Simulation ended without to finish all tasks.  Tasks not ended : "
                                   f"{list(map(str, remaining))}" + Style.RESET_ALL)
        return self.time, self.energy
