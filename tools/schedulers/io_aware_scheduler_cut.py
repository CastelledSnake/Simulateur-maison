from math import inf
from bisect import bisect_left
from typing import List, Tuple

from io_aware_scheduler import IOAwareScheduler
from simulation.model import Model
from simulation.schedule_order import ScheduleOrder, Order
from tools.tasks.task import Task


class IOAwareSchedulerCut(IOAwareScheduler):
    def __init__(self, name: str, model: Model, scoring_function, bounds_function):
        super().__init__(name, model, scoring_function)
        self.bounds_function = bounds_function

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
        for tested_time in insert_times:
            times, energies = self.light_sim_test(task, tested_time, resources)
            score = self.scoring_function(times, energies)
            if score < best_score:
                best_score, best_time = score, tested_time
        return {ScheduleOrder(order=Order.START_TASK,
                              time=best_time,
                              task=task,
                              nodes=resources)}
