import yaml
import numpy as np
from random import Random
from matplotlib import pyplot as plt
from os.path import dirname, exists, isfile, join
from sys import exit
from datetime import datetime

from simulation.model import Model
from simulation.trace_generator import trace_generator
from tools.node import Node
from tools.schedulers.fifo_scheduler import NaiveScheduler
from tools.storage.hdd_storage_tier import HDD
from tools.storage.ssd_storage_tier import SSD
from tools.tasks.io_task_step import IOTaskStep
from tools.tasks.compute_task_step import ComputeTaskStep
from tools.tasks.task import Task
from tools.storage.file import File
from __main__ import time_energy_graph_generation

if __name__ == "__main__":
    #########################
    # Cluster configuration #
    #########################

    nodes_config_path = join(dirname(__file__), "nodes.yaml")
    simulation_config_path = join(dirname(__file__), "simulation.yaml")
    hdds_config_path = join(dirname(__file__), "hdds.yaml")
    ssds_config_path = join(dirname(__file__), "ssds.yaml")
    files_config_path = join(dirname(__file__), "files.yaml")

    # Loading the nodes from config for our cluster simulation
    if not exists(nodes_config_path) or not isfile(nodes_config_path):
        nodes_core_count = [64] * 2 + [128] * 4
        nodes = {f'Mirabelle_{k}': {"max_frequency": 5000000000,
                                    "min_frequency": 1000000000,
                                    "core_count": core_count,
                                    "static_power": float(core_count),
                                    "sleep_power": 5.,
                                    "coefficient_dynamic_power": 1e-28,
                                    "coefficient_leakage_power": 1e-10}
                 for k, core_count in enumerate(nodes_core_count)}
        with open(nodes_config_path, "w") as file:
            file.write(yaml.dump(nodes))
    with open(nodes_config_path, "r") as file:
        nodes_config = yaml.load(file, yaml.Loader)
        for expected_field, field_type in [("max_frequency", int), ("min_frequency", int), ("core_count", int),
                                           ("static_power", float), ("sleep_power", float),
                                           ("coefficient_dynamic_power", float), ("coefficient_leakage_power", float)]:
            if not all([expected_field in node_config for node_config in nodes_config.values()]):
                raise RuntimeError(f'Missing field "{expected_field}" of type {field_type} in config file '
                                   f'"{nodes_config_path}".')
        nodes = [Node(name=node_name,
                      max_frequency=node_config["max_frequency"],
                      min_frequency=node_config["min_frequency"],
                      core_count=node_config["core_count"],
                      static_power=node_config["static_power"],
                      sleep_power=node_config["sleep_power"],
                      coefficient_dynamic_power=node_config["coefficient_dynamic_power"],
                      coefficient_leakage_power=node_config["coefficient_leakage_power"])
                 for node_name, node_config in nodes_config.items()]
    print("Nodes: " + ["Empty", "\n  - " + "\n  - ".join(map(str, nodes))][len(nodes) > 0], end="\n\n")

    # Creating the HDD for our cluster simulation. TODO: all values have been generated with the pifometric method.
    if not exists(hdds_config_path) or not isfile(hdds_config_path):
        hdds = {f'HDD_{k}': {"capacity": int(2e12),
                             "throughput": int(500e6),
                             "latency": 10e-3,
                             "disk_radius": 0.1,
                             "disk_max_spin": 10000 * np.pi,
                             "disk_min_spin": 1000 * np.pi,
                             "arm_mass": 1e-3,
                             "arm_radius": 0.1,
                             "arm_max_spin": 1e-2,
                             "electronic_power": 0.5,
                             "sleep_power": 1.}
                for k in enumerate([1])}
        with open(hdds_config_path, "w") as file:
            file.write(yaml.dump(hdds))
    with open(hdds_config_path, "r") as file:
        hdds_config = yaml.load(file, yaml.Loader)
        for expected_field, field_type in [("capacity", int), ("throughput", int), ("latency", float),
                                           ("disk_radius", float), ("disk_max_spin", float),
                                           ("disk_min_spin", float), ("arm_mass", float), ("arm_radius", float),
                                           ("arm_max_spin", float), ("electronic_power", float),
                                           ("sleep_power", float)]:
            if not all([expected_field in hdd_config for hdd_config in hdds_config.values()]):
                raise RuntimeError(f'Missing field "{expected_field}" of type {field_type} in config file '
                                   f'"{hdds_config_path}".')
        hdds = [HDD(name=hdd_name, capacity=hdd_config["capacity"], throughput=hdd_config["throughput"],
                    latency=hdd_config["latency"], disk_radius=hdd_config["disk_radius"],
                    disk_max_spin=hdd_config["disk_max_spin"], disk_min_spin=hdd_config["disk_min_spin"],
                    arm_mass=hdd_config["arm_mass"], arm_radius=hdd_config["arm_radius"],
                    arm_max_spin=hdd_config["arm_max_spin"],
                    electronic_power=hdd_config["electronic_power"], sleep_power=hdd_config["sleep_power"])
                for hdd_name, hdd_config in hdds_config.items()]

    # Creating the SSD for our cluster simulation. TODO: all values have been generated with the pifometric method.
    if not exists(ssds_config_path) or not isfile(ssds_config_path):
        ssds = {f'SSD_{k}': {"capacity": int(2e12),
                             "throughput": int(500e6),
                             "latency": 10e-3,
                             "max_read_power": 10.,
                             "max_write_power": 13.,
                             "leakage_power": 2.,
                             "sleep_power": 1.}
                for k in enumerate([1])}
        with open(ssds_config_path, "w") as file:
            file.write(yaml.dump(ssds))
    with open(ssds_config_path, "r") as file:
        ssds_config = yaml.load(file, yaml.Loader)
        for expected_field, field_type in [("capacity", int), ("throughput", int), ("latency", float),
                                           ("max_read_power", float), ("max_write_power", float),
                                           ("leakage_power", float), ("sleep_power", float)]:
            if not all([expected_field in ssd_config for ssd_config in ssds_config.values()]):
                raise RuntimeError(f'Missing field "{expected_field}" of type {field_type} in config file '
                                   f'"{ssds_config_path}".')
        ssds = [SSD(name=ssd_name, capacity=ssd_config["capacity"], throughput=ssd_config["throughput"],
                    latency=ssd_config["latency"], max_read_power=ssd_config["max_read_power"],
                    max_write_power=ssd_config["max_write_power"], leakage_power=ssd_config["leakage_power"],
                    sleep_power=ssd_config["sleep_power"])
                for ssd_name, ssd_config in ssds_config.items()]

    # Creation of the global storage list for the simulation.
    storage = hdds + ssds
    # print("Storage: \n  -"+"\n  -".join(map(str, storage))+"\n")

    ####################
    # Trace generation #
    ####################

    trace_path = join(dirname(__file__), "trace.yaml")
    if not exists(trace_path) or not isfile(trace_path):
        rand = Random(123)
        trace = trace_generator(nodes, storage, 42, 10, rand, preferred_storage=hdds[0])
        with open(trace_path, "w") as file:
            file.write(yaml.dump({task.name: {"task_submission_timestamp": task_submission_timestamp,
                                              "min_thread_count": task.min_thread_count,
                                              "steps": [{"step_type": type(step).__name__,
                                                         "flop": step.flop}
                                                        if type(step) is ComputeTaskStep
                                                        else {"file": step.file.name,
                                                              "step_type": type(step).__name__,
                                                              "total_io_volume": step.total_io_volume,
                                                              "average_io_size": step.average_io_size,
                                                              "io_pattern": type(step.io_pattern).__name__} for step in
                                                        task.steps],
                                              "dependencies": list(map(lambda task: task.name, task.dependencies))}
                                  for task, task_submission_timestamp in trace.tasks_ts}))
    with open(trace_path, "r") as file:
        task_ts = [(Task(name=task_name,
                         steps=[ComputeTaskStep(task_data["flop"])
                                if task_data["step_type"] == ComputeTaskStep.__name__
                                else IOTaskStep(files[task_data["file"]],
                                                task_data["total_io_volume"],
                                                task_data["average_io_size"],
                                                task_data["io_pattern"])
                                for step_data in task_data["steps"]],
                         min_thread_count=task_data["min_thread_count"],
                         dependencies=None), task_data["task_submission_timestamp"])
                   for task_name, task_data in yaml.load(file, yaml.Loader)]

    exit(0)

    # Creating a fixed-seed random generator for trace generation
    print("Trace:\n  -" + "\n  -".join([f'At timestamp {timestamp}: {trace}' for trace, timestamp in trace.tasks_ts]))

    # And at last, randomly initializing the initial state of the different storage tiers and file
    # files preferences are respected as much as possible
    files = list(set([step.file
                      for task in tuple(zip(*trace.tasks_ts))[0]
                      for step in task.steps if type(step) is IOTaskStep]))
    rand.shuffle(files)
    for file in files:
        if file.preferred_storage.occupation + file.size <= file.preferred_storage.capacity:
            file.storage = file.preferred_storage
    for file in files:
        if file.preferred_storage.capacity - file.preferred_storage.occupation >= file.size:
            if file not in file.preferred_storage.files:
                file.preferred_storage.files.add(file)
                file.preferred_storage.occupation += file.size
        else:
            if file.preferred_storage in hdds:
                available_media = [media for media in hdds if media.capacity - media.occupation >= file.size]
            else:
                available_media = [media for media in ssds if media.capacity - media.occupation >= file.size]
            if len(available_media) == 0:
                available_media = [media for media in ssds + hdds if
                                   media.capacity - media.occupation >= file.size]
            if len(available_media) == 0:
                raise RuntimeError(f'The following file could not fit in any storage media: {file}')
            if file not in rand.choice(available_media).files:
                rand.choice(available_media).files.add(file)
                rand.choice(available_media).occupation += file.size
    # print("Files:\n  -"+"\n  - ".join(map(str, sorted(files, key=lambda file: file.name))))

    #######################
    # Scheduler selection #
    #######################

    # Creating a new fixed-seed random generator for the scheduler itself
    rand = Random(789)

    # Create the scheduler
    scheduler = NaiveScheduler("Naive_Scheduler_v1", nodes, storage, rand)

    ####################################
    # Simulation & result presentation #
    ####################################

    # Initialize the simulation
    simulation = Simulation(nodes, [hdd], [ssd], files, trace, scheduler)

    # Run the simulation
    times, energies = simulation.simulate()

    # Print the results
    time_energy_power_graph_generation(times, energies)
