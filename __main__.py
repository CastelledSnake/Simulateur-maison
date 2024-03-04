import numpy as np
from random import Random
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from typing import List

from colorama import Style, Fore, Back

from simulation.model import Model
from simulation.trace_generator import trace_generator
from simulation.files_generator import files_generator
from tools.node import Node
from tools.schedulers.fifo_scheduler import NaiveScheduler
from tools.schedulers.io_aware_scheduler import IOAwareScheduler
from tools.schedulers.io_aware_scheduler_cut import IOAwareSchedulerCut
from tools.storage.hdd_storage_tier import HDD
from tools.storage.ssd_storage_tier import SSD

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


def linear_race(tested: float, limit: float, k: int, n: int):
    """
    Returns True iff the tested value over the limit value is out of boundaries, according to an affine function.
    :param tested: A value to place as numerator.
    :param limit: A value to place as denominator.
    :param k: A value associated with the variable in an affine equation.
    :param n: A value associated with the multiplicative coefficient in an affine equation.
    :return: True iff the tested value over the limit value is out of boundaries, according to an affine function.
    """
    return tested/limit >= 2 - k/(n+1)


def exponential_race(tested: float, limit: float, k: int, n: int):
    """
        Returns True iff the tested value over the limit value is out of boundaries, according to an exponential function.
        :param tested: A value to place as numerator.
        :param limit: A value to place as denominator.
        :param k: A value associated with the variable in an exponential equation.
        :param n: A value associated with the multiplicative coefficient in an exponential equation.
        :return: True iff the tested value over the limit value is out of boundaries, according to an exponential function.
        """
    return tested/limit >= 1 + np.exp(-1/(n+1-k))


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
    metrics_naive: List = []
    metrics_aware_normal: List = []
    metrics_aware_linear: List = []
    metrics_aware_exponential: List = []
    timelapses_naive: List[timedelta] = []
    timelapses_aware_normal: List[timedelta] = []
    timelapses_aware_linear: List[timedelta] = []
    timelapses_aware_exponential: List[timedelta] = []
    seeds = range(1)
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
        trace_for_aware_normal = trace_generator(list_nodes=nodes,
                                                 list_storage=storage,
                                                 sample_size=42,
                                                 mean_task_size=10,
                                                 files=files,
                                                 rng=Random(seed))
        trace_for_aware_linear = trace_generator(list_nodes=nodes,
                                                 list_storage=storage,
                                                 sample_size=42,
                                                 mean_task_size=10,
                                                 files=files,
                                                 rng=Random(seed))
        trace_for_aware_exponential = trace_generator(list_nodes=nodes,
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
        model_for_aware_normal = Model(nodes=nodes,
                                       hdds=[hdd],
                                       ssds=[ssd],
                                       list_files=list(files),
                                       tasks_trace=trace_for_aware_normal)
        model_for_aware_linear = Model(nodes=nodes,
                                       hdds=[hdd],
                                       ssds=[ssd],
                                       list_files=list(files),
                                       tasks_trace=trace_for_aware_linear)
        model_for_aware_exponential = Model(nodes=nodes,
                                            hdds=[hdd],
                                            ssds=[ssd],
                                            list_files=list(files),
                                            tasks_trace=trace_for_aware_exponential)

        ########################
        # Scheduler generation #
        ########################

        # Creating a new fixed-seed random generator for the scheduler itself
        # rand_scheduler = Random(1)
        # For now, schedulers don't care about randomness.

        # Create the schedulers
        naive_scheduler = NaiveScheduler(name="NaiveScheduler_seed=" + str(seed),
                                         model=model_for_naive,
                                         scoring_function=delay_scoring)
        aware_normal_scheduler = IOAwareScheduler(name="IOAwareScheduler_Normal_v2.0_seed=" + str(seed),
                                                  model=model_for_aware_normal,
                                                  scoring_function=scoring_function)
        aware_linear_scheduler = IOAwareSchedulerCut(name="IOAwareSchedulerLinear_v2.0_seed=" + str(seed),
                                                     model=model_for_aware_linear,
                                                     scoring_function=scoring_function,
                                                     boundaries_function=linear_race)
        aware_exponential_scheduler = IOAwareSchedulerCut(name="IOAwareSchedulerExponential_v2.0_seed=" + str(seed),
                                                          model=model_for_aware_exponential,
                                                          scoring_function=scoring_function,
                                                          boundaries_function=exponential_race)

        ########################
        # Simulation execution #
        ########################

        # Run the simulation
        t0 = datetime.now()  # The time for formatted record folder name.
        # A previous version was using the syntax '''%Y-%m-%d_à_%H-%M'-%S"''' for the date format.
        #times_naive, energies_naive = model_for_naive.simulate(
            #record_folder=f"enregistrements_automatiques/{naive_scheduler.name}/résultats_du_"
            #              + t0.strftime('''%Y-%m-%d_à_%H-%M-%S'''),
            #scheduler=naive_scheduler)
        t1 = datetime.now()
        #times_aware_normal, energies_aware_normal = model_for_aware_normal.simulate(
        #    record_folder=f"enregistrements_automatiques/{aware_normal_scheduler.name}/résultats_du_"
        #                  + t0.strftime('''%Y-%m-%d_à_%H-%M-%S'''),
        #    scheduler=aware_normal_scheduler)
        t2 = datetime.now()
        times_aware_linear, energies_aware_linear = model_for_aware_linear.simulate(
            record_folder=f"enregistrements_automatiques/{aware_linear_scheduler.name}/résultats_du_"
                          + t0.strftime('''%Y-%m-%d_à_%H-%M-%S'''),
            scheduler=aware_linear_scheduler)
        t3 = datetime.now()
        times_aware_exponential, energies_aware_exponential = model_for_aware_exponential.simulate(
            record_folder=f"enregistrements_automatiques/{aware_exponential_scheduler.name}/résultats_du_"
                          + t0.strftime('''%Y-%m-%d_à_%H-%M-%S'''),
            scheduler=aware_exponential_scheduler)
        t4 = datetime.now()

        metrics_naive.append(scoring_function(times_naive, energies_naive))
        metrics_aware_normal.append(scoring_function(times_aware_normal, energies_aware_normal))
        metrics_aware_linear.append(scoring_function(times_aware_linear, energies_aware_linear))
        metrics_aware_exponential.append(scoring_function(times_aware_exponential, energies_aware_exponential))
        timelapses_naive.append(t1 - t0)
        timelapses_aware_normal.append(t2 - t1)
        timelapses_aware_linear.append(t3 - t2)
        timelapses_aware_exponential.append(t4 - t3)

        ##########################
        # Final graph generation #
        ##########################
        # time_energy_power_graph_generation(times, energies)

    positive_normal, negative_normal, positive_linear, negative_linear, positive_exponential, negative_exponential = 0, 0, 0, 0, 0, 0
    time_normal, time_linear, time_exponential = 0, 0, 0
    for k in range(len(metrics_naive)):
        # Check if the aware (normal) scheduler produced a better result than the naive scheduler
        if metrics_naive[k] > metrics_aware_normal[k]:
            positive_normal += 1
        else:
            negative_normal += 1
        # Check if the linear scheduler produced a better result than the normal scheduler
        if metrics_aware_normal[k] > metrics_aware_linear[k]:
            positive_linear += 1
        else:
            negative_linear += 1
        # Check if the exponential scheduler produced a better result than the normal scheduler
        if metrics_aware_normal[k] > metrics_aware_exponential[k]:
            positive_exponential += 1
        else:
            negative_exponential += 1

        # Check if the aware (normal) scheduler was faster than the naive scheduler
        if timelapses_naive[k] > timelapses_aware_normal[k]:
            time_normal += 1
        # Check if the linear scheduler was faster than the normal scheduler
        if timelapses_aware_normal[k] > timelapses_aware_linear[k]:
            time_linear += 1
        # Check if the exponential scheduler was faster than the normal scheduler
        if timelapses_aware_normal[k] > timelapses_aware_exponential[k]:
            time_exponential += 1

    print(f"IOAwareSchedulerNormal was better than FIFO in {positive_normal}/{len(seeds)}, and worst or equal in {negative_normal}/{len(seeds)}.")
    print(f"    IOAwareSchedulerNormal was faster than FIFO in {time_normal}/{len(seeds)}.")
    print(f"IOAwareSchedulerLinear was better than Normal in {positive_linear}/{len(seeds)}, and worst or equal in {negative_linear}/{len(seeds)}.")
    print(f"    IOAwareSchedulerLinear was faster than Normal in {time_linear}/{len(seeds)}.")
    print(f"IOAwareSchedulerExponential was better than Normal in {positive_exponential}/{len(seeds)}, and worst or equal in {negative_exponential}/{len(seeds)}.")
    print(f"    IOAwareSchedulerExponential was faster than Normal in {time_exponential}/{len(seeds)}.")
    print(Fore.RED + f"Naive VS Aware VS Linear VS Exponential\n    Naive VS Aware VS Linear VS Exponential" + Style.RESET_ALL)
    for k in range(len(seeds)):
        print(f"{metrics_naive[k]} VS {metrics_aware_normal[k]} VS {metrics_aware_linear[k]} VS {metrics_aware_exponential[k]}")
        print(f"    {timelapses_naive[k]} VS {timelapses_aware_normal[k]} VS {timelapses_aware_linear[k]} VS {timelapses_aware_exponential[k]}")


    # TODO : Ensemble de quelques tâches simples pour test des politiques d'ordonnancement.
