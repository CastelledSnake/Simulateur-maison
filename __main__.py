import numpy as np
from random import Random
from matplotlib import pyplot as plt
from datetime import datetime
from typing import List

from simulation.model import Model
from simulation.trace_generator import trace_generator
from simulation.files_generator import files_generator
from simulation.trace import TaskSchedulingTrace
from tools.node import Node
from tools.schedulers.fifo_scheduler import NaiveScheduler
from tools.schedulers.io_aware_scheduler import IOAwareScheduler
from tools.storage.hdd_storage_tier import HDD
from tools.storage.ssd_storage_tier import SSD
from tools.tasks.io_task_step import IOTaskStep

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


def time_energy_power_graph_generation(times: List[float], energies: List[float]):
    """
    Plots a graph of energy and power as a function of time.
    It requires a list of moments and a list of energies fitting these times. Power is calculated by deriving
    energy with respect to time.
    :param times: A list of floats, representing times.
    :param energies: A list of floats, representing energy consumption at each time from argument 'times'
    :return: Shows a graph.
    """
    if len(times) != len(energies):
        raise ValueError(f"'times' has length {len(times)} and 'energy' has length {len(energies)}. "
                         f"They must be the same.")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(times, energies, 'r-*', label="Énergie consommée")
    ax2.plot(times, np.gradient(energies, times), 'b-', label="Puissance requise")
    ax2.set_ylim(ymin=0)

    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Énergie (J)', color='r')
    ax2.set_ylabel('Puissance (W)', color='b')

    plt.show()

    #################################
    # Scoring functions definitions #
    #################################


def delay_scoring(delay: List[float], energy: List[float]):
    """
    Scoring function only dependent on delay.
    :param delay: float representing the time taken by one or more Tasks to complete.
    :return: delay
    """
    return delay[-1]


def energy_scoring(delay: List[float], energy: List[float]):
    """
    Scoring function only dependent on energy.
    :param energy: float representing the energy taken by the system to complete one or more Tasks.
    :return: energy
    """
    return energy[-1]


def edp(delay: List[float], energy: List[float]):
    """
    Energy-Delay-Product scoring function. Computes a time-energy trade-off by multiplying delay and energy taken
    by one or more Tasks to complete on the system.
    :param delay:
    :param energy:
    :return:
    """
    return energy[-1] * delay[-1]


def ed2p(delay: List[float], energy: List[float]):
    """
    Energy-Delay-Squared-Product scoring function. Computes a time-energy trade-off by multiplying energy and
    squared delay taken by one or more Tasks to complete on the system.
    :param delay:
    :param energy:
    :return:
    """
    return energy[-1] * delay[-1] ** 2


if __name__ == "__main__":

    #########################
    # Cluster configuration #
    #########################

    # Creating the nodes for our cluster simulation. TODO: all values have been generated with the pifometric method.
    nodes = [Node(name=f'Mirabelle_{i}',
                  max_frequency=5000000000,
                  min_frequency=1000000000,
                  frequencies=[1000000000, 1500000000, 2000000000, 2500000000, 3000000000, 3500000000, 4000000000,
                               4500000000, 5000000000],
                  core_count=core_count,
                  static_power=float(core_count),  # 1W per core for static power
                  sleep_power=5.,
                  coefficient_dynamic_power=1e-28,
                  coefficient_leakage_power=1e-10)
             for i, core_count in enumerate([64] * 2 + [128] * 4)]
    print("Nodes: " + ["Empty", "\n  - " + "\n  - ".join(map(str, nodes))][len(nodes) > 0], end="\n\n")
    # Creating the storage for our cluster simulation. TODO: all values have been generated with the pifometric method.
    hdd = HDD(name="HDD",
              capacity=int(2e12),
              throughput=int(500e6),
              latency=10e-3,
              disk_radius=0.1,
              disk_max_spin=10000 * np.pi,
              disk_min_spin=1000 * np.pi,
              arm_mass=1e-3,
              arm_radius=0.1,
              arm_max_spin=1e-2 * np.pi,
              electronic_power=0.5,
              sleep_power=1.)
    hdds = [hdd]
    ssd = SSD(name="SSD",
              capacity=int(2e12),
              throughput=int(500e6),
              latency=10e-6,
              max_read_power=10.,
              max_write_power=13.,
              leakage_power=2.,
              sleep_power=1.)
    ssds = [ssd]

    storage = hdds + ssds
    print("Storage: \n  -" + "\n  -".join(map(str, storage)) + "\n")

    ####################
    # Files generation #
    ####################

    # Creating a fixed-seed random generator for trace generation
    rand_files = Random(20100)

    # Create the Files
    files = files_generator(hdds=hdds,
                            ssds=ssds,
                            number_of_files=123,
                            min_file_size=int(10e5),
                            max_file_size=int(10e9),
                            rng=rand_files)

    # And at last, randomly initializing the initial state of the different storage tiers and file
    # files preferences are respected as much as possible
    # Try to allocate Files to their preferred Storage. By default, we assume that Storages have enough space to store
    # all their Files, and that counterexamples are rare.
    for file in files:
        if file.preferred_storage.occupation + file.size <= file.preferred_storage.capacity:
            file.storage = file.preferred_storage
    # Dealing with counterexamples
    print("Files:\n  -" + "\n  - ".join(map(str, sorted(files, key=lambda my_file: my_file.name))) + "\n")

    #####
    ##########
    ####################
    ########################################
    # SIMULATIONS GENERATION AND EXECUTION #
    ########################################
    ####################
    ##########
    #####

    scoring_function = edp
    metrics_naive: List[float] = []
    metrics_aware: List[float] = []
    seeds = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12]
    for seed in seeds:
        ####################
        # Trace generation #
        ####################
        trace_for_naive = trace_generator(list_nodes=nodes,
                                          list_storage=storage,
                                          sample_size=42,
                                          mean_task_size=10,
                                          files=files,
                                          rng=Random(seed))
        trace_for_aware = trace_generator(list_nodes=nodes,
                                          list_storage=storage,
                                          sample_size=42,
                                          mean_task_size=10,
                                          files=files,
                                          rng=Random(seed))
        # print("Trace:\n  -" + "\n  -".join(
        #    [f'At timestamp {timestamp}: {repr(task)}' for task, timestamp in trace.tasks_ts]) + "\n")

        ###############################
        # Simulation model generation #
        ###############################

        # Initialize the simulation model
        model_for_naive = Model(nodes=nodes,
                                hdds=[hdd],
                                ssds=[ssd],
                                list_files=list(files),
                                tasks_trace=trace_for_naive)
        model_for_aware = Model(nodes=nodes,
                                hdds=[hdd],
                                ssds=[ssd],
                                list_files=list(files),
                                tasks_trace=trace_for_aware)

        ########################
        # Scheduler generation #
        ########################

        # Creating a new fixed-seed random generator for the scheduler itself
        # rand_scheduler = Random(1)
        # For now, schedulers don't care about randomness.

        # Create the schedulers
        naive_scheduler = NaiveScheduler("NaiveScheduler_seed=" + str(seed), model_for_naive)
        aware_scheduler = IOAwareScheduler("IOAwareScheduler_v1.0_seed=" + str(seed), model_for_aware, scoring_function)

        ########################
        # Simulation execution #
        ########################

        # Run the simulation
        now = datetime.now()
        times_naive, energies_naive = model_for_naive.simulate(
            record_folder=f"enregistrements_automatiques/{naive_scheduler.name}/résultats_du_"
                          + now.strftime('''%Y-%m-%d_à_%H-%M'-%S"'''),
            scheduler=naive_scheduler)
        times_aware, energies_aware = model_for_aware.simulate(
            record_folder=f"enregistrements_automatiques/{aware_scheduler.name}/résultats_du_"
                          + now.strftime('''%Y-%m-%d_à_%H-%M'-%S"'''),
            scheduler=aware_scheduler)

        metrics_naive.append(scoring_function(times_naive, energies_naive))
        metrics_aware.append(scoring_function(times_aware, energies_aware))
        ##########################
        # Final graph generation #
        ##########################
        # time_energy_power_graph_generation(times, energies)

    positive, negative = 0, 0
    for k in range(len(metrics_naive)):
        if metrics_naive[k] > metrics_aware[k]:
            positive += 1
        else:
            negative += 1

    print(f"IOAwareScheduler was better in {positive}/{len(seeds)}, and worst or equal in {negative}/{len(seeds)}.")
    for k in range(len(seeds)):
        print(f"{metrics_naive[k]} VS {metrics_aware[k]}")

    # TODO : Ensemble de quelques tâches simples pour test des politiques d'ordonnancement.
