# from random import Random
from math import inf
from typing import List, Tuple, Dict, Set
from copy import *
from bisect import bisect_left

from simulation.schedule_order import ScheduleOrder, Order
from simulation.next_event import Event, NextEvent
from simulation.model import Model
from tools.node import Node
from tools.schedulers.abstract_scheduler import AbstractScheduler
# from tools.storage.abstract_storage_tier import Storage
from tools.tasks.task import Task, State


class IOAwareScheduler(AbstractScheduler):
    """
    This IO-aware scheduler works as follow :  # TODO Traduire en Anglais
    L’ordonnanceur commence par exécuter la tâche 0, qui arrive à l’instant 0.
    Tant que toutes les tâches n’ont pas été exécutées :
        Lorsqu’une nouvelle tâche N est soumise à l’ordonnanceur :
            L’ordonnanceur détermine le temps tf et l’énergie nécessaires pour exécuter le paquet de tâches déjà en
                cours d’exécution.
            S’il n’y a pas assez de cœurs présentement disponibles pour exécuter la tâche n, ou si les dépendances de
                la tâche N ne sont pas encore terminées, l’ordonnanceur cherche le moment td à compter duquel il y aura
                assez de cœurs, et où les dépendances seront toutes finies. Si les conditions sont déjà remplies, td
                prend la valeur de l’instant présent.
            L’ordonnanceur recalcule le temps et l’énergie nécessaires à la réalisation du paquet si on venait y
                insérer la tâche N à différents instants entre td et tf (usuellement, ces ‘instants’ isolés
                correspondront aux fins de phases de chacune des tâches déjà en cours d’exécution).
            L’ordonnanceur adopte, pour la suite de la simulation, la solution évaluée qui présente le coût le plus
                faible suivant la métrique d’évaluation (la ‘métrique’ pouvant être l’énergie seule, le temps seul, le
                produit de l’énergie et du temps (Energy-Delay-Product) ou encore le produit de l’énergie et du carré
                du temps (Energy-Delay-Squared-Product), chacune de ces métriques pouvant faire l’objet d’un
                ordonnanceur particulier).
    """

    def __init__(self, name: str, model: Model, scoring_function):
        """
        Constructor of the IOAwareScheduler class.
        :param name: String to name the Scheduler.
        :param model:
        :param scoring_function:
        """
        # List containing all the available resources :
        AbstractScheduler.__init__(self, name, model, scoring_function)
        self.pre_allocated_cores: Dict[Node: Dict[Task: int]] = {}  # When a Task arrives, it is given a timeslot, thus
        # it occupies some cores on the Nodes. This structure should avoid the scheduler to give such pre-allocated
        # cores to future incoming Tasks.
        for node in self.model.nodes:
            self.pre_allocated_cores[node] = {None: 0}
        # If no scoring function is passed, raises an error : this scheduler needs an explicit one.
        if not self.scoring_function:
            raise ValueError(f"IOAwareScheduler requires a scoring function. None was given.")

    def __str__(self):
        return self.name

    def model_deep_copy(self, main_order: ScheduleOrder = None, secondary_orders: Set[ScheduleOrder] = None) -> Model:
        """
        Creates a deep copy of scheduler's model in which all non-submitted Tasks are erased (a light_model),
        except for the one in the optional argument new_event, if any.
        :param main_order: Optional ScheduleOrder, stating when and on which resources, a Task should be launched.
        :param secondary_orders:
        :return: The light Model.
        """
        # TODO Pourquoi les ScheduleOrders ne se transmettent pas d'un Model à un light_model ?
        light_model = deepcopy(self.model)
        task_copy: Task or None = None
        # Insures that all Tasks this light simulation shall take into account are in its tasks_trace, and no other.
        for task_t in reversed(light_model.tasks_trace.tasks_ts):
            if main_order and (task_t[0].name == main_order.task.name):
                # If the Task is the one to be tested, we ensure that we pass the copied version of the Task to the main
                # ScheduleOrder, not the original one, from the main Model. This is the purpose of task_copy.
                task_copy = task_t[0]
            # Any Task whose name is not in the ScheduleOrders given in argument and which is not executing/finished is
            # not desired in the light simulation.
            if task_t[0].state in (State.NOT_SUBMITTED, State.SUBMITTED) \
                    and (not main_order or task_t[0].name != main_order.task.name):
                sec_del: bool = True
                for schedule_order in secondary_orders:
                    if task_t[0].name == schedule_order.task.name:
                        sec_del = False
                if sec_del:
                    light_model.tasks_trace.tasks_ts.remove(task_t)
        for schedule_order in self.model.next_orders:
            for schedule_order_copy in light_model.next_orders:
                assert schedule_order.task is not schedule_order_copy.task
        if main_order:
            main_order.task = task_copy
            light_model.next_orders.add(main_order)  # To avoid the case 'light_model.next_orders = {None}'
        else:  # If no main_order, the newly submitted Task was actually removed from the light model.
            light_model.tasks_trace.lsti -= 1
        return light_model

    @staticmethod
    def simulation_overview(light_simulation, new_order: ScheduleOrder = None):
        """
        Executes the entire simulation considering no new Task is to be submitted, except the one given in new_order,
        if any. According to these thesis, computes, time and energy that this light simulation will experience.
        :param light_simulation: The light simulation on which we want an overview.
        :param new_order: Optional argument : if the user wants the overview to begin one new Task, it can do so
        by giving an appropriated ScheduleOrder.
        :return: Two lists of floats : 1st is the list of times identified by the overview, and 2nd is the associated
        list of energy consumption values (each taken from time=0) planned by the overview.
        """
        assert (new_order is None) or (new_order.order == Order.START_TASK)
        if new_order:
            light_simulation.next_orders.update({new_order})
            (times, energies) = light_simulation.simulate(scheduler=None, record_folder="")
            return times, energies
        else:
            return light_simulation.simulate(scheduler=None, record_folder="")

    def find_resources(self, task: Task) -> Tuple[List[Tuple[Node, int]], float]:
        """
        Get the soonest moment such as the simulation will be able to execute the task in parameter, and get the cores
        on which this Task shall be executed. The soonest moment verifies the following conditions :
        - Enough cores can be booked for the task.
        - All Task's dependencies are finished.
        :param task: The task to plan the execution for.
        :return: The list of couples (node, cores) to be allocated to the task, and the soonest moment as a float.
        """
        [av_nodes, tot_av_cores] = self.sort_least_used_nodes(task)
        m_cores = task.min_thread_count  # We assume that one thread occupies exactly one core.
        exe: bool = self.all_dependencies_check(task) and tot_av_cores >= m_cores
        # exe doesn't tell anything concerning I/O bandwidth, because we want to be able to create I/O contention.
        if exe:
            # If exe == True, there is no need to copy the Model to evaluate time_begin : the Task can be executed now.
            time_begin = self.model.time[-1]
            reserved_nodes = self.book_least_used_cores(task, av_nodes)
            return reserved_nodes, time_begin

        # The worst case : we need to iterate over next simulation events to find the appropriate moment and resources.
        light_model = self.model_deep_copy(main_order=None,
                                           secondary_orders=self.model.next_orders)
        light_model.next_event = NextEvent(events={Event.SIMULATION_START},
                                           task=None,
                                           time=0.,
                                           order=None)
        while not exe:
            while (Event.TASK_END not in light_model.next_event.events) \
                    and (Event.SIMULATION_TERMINAISON not in light_model.next_event.events):
                light_model.simulation_step()
            exe = (self.all_dependencies_check(task, light_model)) and (tot_av_cores >= m_cores)
            if not exe:
                raise OverflowError(f"Impossible to execute {task} on system {self.model.nodes}.")
        time_begin = light_model.time[-1]
        reserved_nodes = self.book_least_used_cores(task, av_nodes)
        return reserved_nodes, time_begin

    def light_sim_test(self, task: Task = None, time_begin: float = None, resources: List[Tuple[Node, int]] = None):
        """
        Executes the light simulation related to the submission of a Task at time_begin, on given resources and return
        metrics. If one of these parameters is not given, returns the metrics of the base light simulation
        :param task:
        :param time_begin:
        :param resources:
        :return:
        """
        if task and (time_begin is not None) and resources:  # If all optional parameters are filled, this light
            # simulation shall try to execute a Task.
            new_order: ScheduleOrder = ScheduleOrder(order=Order.START_TASK,
                                                     time=time_begin,
                                                     task=task,
                                                     nodes=resources)
            secondary_orders = self.model.next_orders.copy()
            light_model: Model = self.model_deep_copy(main_order=new_order,
                                                      secondary_orders=secondary_orders)
            return self.simulation_overview(light_model, new_order)
        if (task and time_begin and not resources) \
                or (task and (time_begin is None) and resources) \
                or ((not task) and time_begin and resources) \
                or ((not task) and (time_begin is None) and resources) \
                or ((not task) and time_begin and (not resources)) \
                or (task and (time_begin is None) and (not resources)):
            raise NotImplementedError(f"At least one of the required elements required to execute a new Task "
                                      f"is not given")
        # Otherwise, this light simulation is executed to obtain a baseline.
        light_model_baseline = self.model_deep_copy(main_order=None,
                                                    secondary_orders=self.model.next_orders)
        return self.simulation_overview(light_model_baseline)

    def on_new_task(self, task: "Task"):
        """
        Deals with the arrival of a new task in the queue of candidates.
        :param task: The oncoming task.
        :return: A list of schedule orders
        """
        # Get the base times and energy consumptions if no new Task is submitted, only for evaluating
        # performances afterwards.
        times_baseline, energies_baseline = self.light_sim_test()

        # Get the resources and the soonest moment the Task can be executed.
        resources, time_begin = self.find_resources(task)
        assert time_begin in times_baseline

        # Isolate all the oncoming moments in the baseline simulation : the scheduler will try to execute the new Task
        # at each of these times to see what happens.
        insert_times: List[float] = times_baseline[bisect_left(times_baseline, time_begin):]

        # best_score variable is initialised with infinite value, and reevaluated for each moment in insert_times.
        best_score, best_time = inf, inf
        for tested_time in insert_times:
            times, energies = self.light_sim_test(task, tested_time, resources)
            score = self.scoring_function(times, energies)
            if score < best_score:
                best_score, best_time = score, tested_time
        return {ScheduleOrder(order=Order.START_TASK,
                              time=best_time,
                              task=task,
                              nodes=resources)}

    def on_task_finished(self, task: "Task"):
        """
        Deals with the ending of a task. This IOAwareScheduler doesn't perform any action when a Task is ending.
        :param task: The task that just finished.
        :return: A list of schedule orders
        """
        return {}
