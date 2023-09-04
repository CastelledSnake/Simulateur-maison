import numpy as np
from random import Random
from matplotlib import pyplot as plt

from model.node import Node
from model.schedulers.fifo_scheduler import NaiveScheduler
from model.storage.hdd_storage_tier import HDD
from model.storage.ssd_storage_tier import SSD
from model.tasks.io_task_step import IOTaskStep
from simulation.simulation import Simulation
from simulation.trace_generator import trace_generator
from simulation.files_generator import files_generator

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

if __name__ == "__main__":
    #########################
    # Cluster configuration #
    #########################

    # Creating the nodes for our cluster simulation. TODO: all values have been generated with the pifometric method.
    nodes = [Node(name=f'Mirabelle_{i}',
                  max_frequency=5000000000,
                  min_frequency=1000000000,
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
    print("Storage: \n  -"+"\n  -".join(map(str, storage))+"\n")

    ####################
    # Trace generation #
    ####################

    # Creating a fixed-seed random generator for trace generation
    rand = Random(20100)

    # Create the Files
    files = files_generator(hdds, ssds, 100, int(10e5), int(10e9), rand)

    # Create the trace.
    trace = trace_generator(nodes, storage, 42, 10, files, rand)
    print("Trace:\n  -"+"\n  -".join([f'At timestamp {timestamp}: {repr(task)}' for task, timestamp in trace.tasks_ts]))

    # And at last, randomly initializing the initial state of the different storage tiers and file
    # files preferences are respected as much as possible
    # Try to allocate Files to their preferred Storage. By default, we assume that Storages have enough space to store
    # all their Files, and that counterexamples are rare.
    for file in files:
        if file.preferred_storage.occupation + file.size <= file.preferred_storage.capacity:
            file.storage = file.preferred_storage
    # Dealing with counterexamples
    print("Files:\n  -"+"\n  - ".join(map(str, sorted(files, key=lambda my_file: my_file.name))))

    #######################
    # Scheduler selection #
    #######################

    # Creating a new fixed-seed random generator for the scheduler itself
    # rand = Random(1)
    # NaiveScheduler doesn't care about randomness.

    # Create the scheduler
    scheduler = NaiveScheduler("Naive_Scheduler_v1", nodes, storage)

    ####################################
    # Simulation & result presentation #
    ####################################

    # Initialize the simulation
    simulation = Simulation(nodes, [hdd], [ssd], list(files), trace, scheduler)

    # Run the simulation
    t, energy = simulation.run()

    # Print the results
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, energy, 'r-*', label="Énergie consommée")
    ax2.plot(t, np.gradient(energy, t), 'b-', label="Puissance requise")
    ax2.set_ylim(ymin=0)

    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Énergie (J)', color='r')
    ax2.set_ylabel('Puissance (W)', color='b')

    plt.show()
    # TODO : Set de quelques tâches simples pour test des politiques d'ordonnancement.
