from math import inf
from numpy import mean
from bisect import bisect_left
from typing import List, Tuple, Dict

from tools.schedulers.io_aware_scheduler import IOAwareScheduler
from simulation.model import Model
from simulation.schedule_order import ScheduleOrder, Order
from tools.node import Node
from tools.tasks.task import Task


class IOAwareSchedulerCut(IOAwareScheduler):
    def __init__(self, name: str, model: Model, scoring_function, boundaries_function):
        super().__init__(name, model, scoring_function)
        self.boundaries_function = boundaries_function
        self.limits = ({}, {})

    def check_boundaries(self, tested_task_times: Dict[Task, float], tested_task_energies: Dict[Task, float]):
        """
        Returns True if the tested tasks are within a range, given by self.boundaries_function.
        :param tested_task_times: A dictionary that maps Tasks to their respective makespan.
        :param tested_task_energies: A dictionary that maps Tasks to their respective energy consumptions.
        :return: True if we are out of boundaries, False otherwise.
        """
        limit_times = self.limits[0]
        avg_tested, avg_limits = [], []
        for task in limit_times.keys():
            for tested_task in tested_task_times.keys():
                if tested_task.name == task.name:
                    avg_tested.append(tested_task_times[tested_task])
                    avg_limits.append(limit_times[task])
        return self.boundaries_function(mean(avg_tested), mean(avg_limits), len(tested_task_times), len(limit_times))

    @staticmethod
    def simulation_overview_task_endings(light_simulation, new_order: ScheduleOrder = None):
        """
        Executes an entire simulation considering no new Task is to be submitted, except the one given in new_order,
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
            (times, energies) = light_simulation.simulate_task_endings(record_folder="",
                                                                       scheduler=None)
            return times, energies
        else:
            return light_simulation.simulate_task_endings(record_folder="",
                                                          scheduler=None)

    def simulation_overview_within_boundaries(self, light_simulation, new_order: ScheduleOrder, energy_tolerance: float):
        """
        Executes an entire simulation considering no new Task is to be submitted, except the one given in new_order,
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
            (times, energies) = light_simulation.simulate_within_boundaries(boundaries_function=self.check_boundaries,
                                                                            limits=self.limits,
                                                                            energy_tolerance=energy_tolerance,
                                                                            record_folder="",
                                                                            scheduler=None)
            return times, energies
        else:
            return light_simulation.simulate_within_boundaries(boundaries_function=self.check_boundaries,
                                                               limits=self.limits,
                                                               energy_tolerance=energy_tolerance,
                                                               record_folder="",
                                                               scheduler=None)

    def light_sim_test_task_endings(self, task: Task = None, time_begin: float = None, resources: List[Tuple[Node, int]] = None):
        """
        Executes the light simulation related to the submission of a Task at time_begin, on given resources and return
        metrics. If one of these parameters is not given, returns the metrics of the base light simulation
        :param task: a new Task to be submitted. Can be None if we just want to execute only the already planned Tasks.
        :param time_begin: if any, the time the new Task should begin.
        :param resources: if any, the resources on which the new Task should be executed.
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
            return self.simulation_overview_task_endings(light_simulation=light_model,
                                                         new_order=new_order)
        elif (task and time_begin and not resources) \
                or (task and (time_begin is None) and resources) \
                or ((not task) and time_begin and resources) \
                or ((not task) and (time_begin is None) and resources) \
                or ((not task) and time_begin and (not resources)) \
                or (task and (time_begin is None) and (not resources)):
            raise NotImplementedError(f"At least one of the required elements required to execute a new Task "
                                      f"is not given")
        # Otherwise, this light simulation is executed to obtain a baseline.
        else:
            light_model_baseline = self.model_deep_copy(main_order=None,
                                                        secondary_orders=self.model.next_orders)
            return self.simulation_overview_task_endings(light_simulation=light_model_baseline)

    def light_sim_test_within_boundaries(self, task: Task, time_begin: float, resources: List[Tuple[Node, int]], energy_tolerance: float):
        """
        Executes the light simulation related to the submission of a Task at time_begin, on given resources and return
        metrics. If one of these parameters is not given, returns the metrics of the base light simulation
        :param task: a new Task to be submitted. Can be None if we just want to execute only the already planned Tasks.
        :param time_begin: if any, the time the new Task should begin.
        :param resources: if any, the resources on which the new Task should be executed.
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
            return self.simulation_overview_within_boundaries(light_simulation=light_model,
                                                              new_order=new_order,
                                                              energy_tolerance=energy_tolerance)
        # This method is not meant to be executed to obtain a baseline.
        else:
            raise NotImplementedError(f"At least one of the required elements to execute a new Task is not given")
        # elif (task and time_begin and not resources) \
        #        or (task and (time_begin is None) and resources) \
        #        or ((not task) and time_begin and resources) \
        #        or ((not task) and (time_begin is None) and resources) \
        #        or ((not task) and time_begin and (not resources)) \
        #        or (task and (time_begin is None) and (not resources)):
        #    raise NotImplementedError(f"At least one of the required elements required to execute a new Task "
        #                              f"is not given")
        # else:
        #    light_model_baseline = self.model_deep_copy(main_order=None,
        #                                                secondary_orders=self.model.next_orders)
        #    return self.simulation_overview_within_boundaries(light_simulation=light_model_baseline)

    def on_new_task(self, task: "Task") -> {ScheduleOrder}:
        """
        Deals with the arrival of a new task in the queue of candidates.
        :param task: The oncoming task.
        :return: A list of schedule orders
        """
        # Get the base times and energy consumptions if no new Task is submitted, only for evaluating
        # performances afterward.
        times_baseline, energies_baseline = self.light_sim_test()

        # Get the resources and the soonest moment the Task can be executed.
        resources, time_begin = self.find_resources(task)
        assert time_begin in times_baseline

        # Isolate all the oncoming moments in the baseline simulation : the scheduler will try to execute the new Task
        # at each of these times to see what happens.
        insert_times: List[float] = times_baseline[bisect_left(times_baseline, time_begin):]

        # best_score variable is initialised with infinite value, and reevaluated for each moment in insert_times.
        best_score, best_time = inf, inf
        # Generates the limits for the light simulation.
        self.limits = self.light_sim_test_task_endings(task, insert_times[0], resources)
        energy_tolerance = max(self.limits[1].values())
        for tested_time in insert_times:
            times, energies = self.light_sim_test_within_boundaries(task, tested_time, resources, energy_tolerance)
            score = self.scoring_function(times, energies)
            if score < best_score:
                best_score, best_time = score, tested_time
        return {ScheduleOrder(order=Order.START_TASK,
                              time=best_time,
                              task=task,
                              nodes=resources)}
