from random import Random
import numpy as np
from matplotlib import pyplot as plt
from model.node import Node
from model.schedulers.fifo_scheduler import NaiveScheduler
from model.storage.hdd_storage_tier import HDD
from model.storage.ssd_storage_tier import SSD
from model.tasks.io_task_step import IOTaskStep
from simulation.simulation import Simulation
from simulation.trace_generator import trace_generator

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

    # Creating the nodes for our cluster simulation
    nodes = [Node(name=f'cherry{i}', max_frequency=5000000000, min_frequency=1000000000, core_count=core_count)
             for i, core_count in enumerate([64] * 2 + [128] * 4)]
    # print("Nodes: " + ["Empty", "\n  - " + "\n  - ".join(map(str, nodes))][len(nodes) > 0], end="\n\n")
    # Creating the storage for our cluster simulation. TODO: all values have been generated with the pifometric method.
    hdd = HDD(name="HDD",
              capacity=int(2e12),
              throughput=int(500e6),
              latency=10e-3,
              disk_radius=0.1,
              disk_max_spin=1000*np.pi,
              disk_min_spin=100*np.pi,
              arm_mass=1e-3,
              arm_radius=0.1,
              arm_max_spin=np.pi*1e-2,
              electronic_power=5,
              sleep_power=1)
    hdds = [hdd, ]
    ssd = SSD(name="SSD",
              capacity=int(2e12),
              throughput=int(500e6),
              latency=10e-6,
              max_read_power=10.,
              max_write_power=13,
              leakage_power=2,
              sleep_power=1)
    ssds = [ssd, ]
    storage = hdds + ssds
    # print("Storage: \n  -"+"\n  -".join(map(str, storage))+"\n")

    ####################
    # Trace generation #
    ####################

    # Creating a fixed-seed random generator for trace generation
    rand = Random(123)

    # Create the trace.
    trace = trace_generator(nodes, storage, 42, 10, rand, preferred_storage=hdd)
    # print("Trace:\n  -"+"\n  -".join([f'At timestamp {timestamp}: {trace}' for trace, timestamp in trace.tasks_ts]))

    # And at last, randomly initializing the initial state of the different storage tiers and file
    # files preferences are respected as much as possible
    files = list(set([step.file
                      for task in tuple(zip(*trace.tasks_ts))[0]
                      for step in task.steps if type(step) is IOTaskStep]))
    rand.shuffle(files)
    for file in files:
        if file.preferred_storage.occupation+file.size <= file.preferred_storage.capacity:
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
