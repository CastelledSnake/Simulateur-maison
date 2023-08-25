from math import floor, log10
from random import Random
from typing import List
from model.node import Node
from model.storage.abstract_storage_tier import Storage
from model.storage.file import File
from model.tasks.abstract_task_step import TaskStep
from model.tasks.compute_task_step import ComputeTaskStep
from model.tasks.io_task_step import IOTaskStep, IOPattern
from model.tasks.task import Task
from simulation.trace import TaskSchedulingTrace
from simulation.utils import benford


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


def trace_generator(list_nodes: List[Node], list_storage: List[Storage], sample_size: int, task_size: int, rng: Random,
                    preferred_storage: Storage) -> TaskSchedulingTrace:
    """
    Generates a TaskSchedulingTrace suitable to run on a Simulation.
    :param list_nodes: List of Nodes that will be used in the simulation to which the set of tasks is created for.
    :param list_storage: List of Storage that will be used in the simulation.
    :param sample_size: The number of tasks to generate
    :param task_size: The average amount of TaskSteps per task.
    :param rng: TBD
    :param preferred_storage:
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

    file_count = 0
    for k in range(sample_size):
        steps: List[TaskStep] = []
        number_of_steps = int(rng.gauss(task_size, sigma_task))
        if number_of_steps <= 0:
            number_of_steps = 1
        for kk in range(number_of_steps // 2):  # We create an alternate sequence of IO and Computation steps.
            steps.append(IOTaskStep(file=File(name=f'file_{file_count}.txt',
                                              size=10**rng.randint(0, 10),
                                              preferred_storage=preferred_storage),
                                    total_io_volume=rng.randrange(total_storage_flow),
                                    average_io_size=rng.randrange(total_storage_flow//10, total_storage_flow),
                                    io_pattern=IOPattern.SEQUENTIAL))
            file_count += 1
            steps.append(ComputeTaskStep(flop=rng.randrange(flop)))
        tasks.append(Task(name=f"task_{k}",
                          steps=steps,
                          min_thread_count=rng.randrange(max_cores),    # We assume that a core takes exactly a thread.
                          dependencies=dependencies_generator(tasks, rng)))
        if timestamps:
            timestamps.append(rng.uniform(timestamps[-1], timestamps[-1] + 10))  # CURRENTLY, the next timestamp is a
            # float taken from a uniform distribution over the interval coming from the last timestamp declared, up
            # to the latter plus 5s.
        else:
            timestamps.append(0)
    return TaskSchedulingTrace(tasks, timestamps)
