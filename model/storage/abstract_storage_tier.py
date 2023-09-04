from abc import ABC, abstractmethod
from typing import Set, List

from model.tasks.task import Task
from model.utils import pretty_print


class Storage(ABC):
    """
    Mother class of all storage devices implemented (HDD, SSD, RAM, ...).
    """
    # On a 2 niveaux de mémoire : HDD et SSD.
    # Le SSD a une latence faible, mais un débit d'IO faible, quand le HDD a le contraire. (cf. nombres d'état de l'art)
    # NB : voir à partir de quelle quantité d'I/O par ordre d'I/O, le HDD devient plus rentable que le SSD.
    #
    # On implémentera la classe "File", dont chaque instance décrit un fichier sur le système.
    # Chaque File a une empreinte mémoire (variable) et un tier de stockage de prédilection.
    # Toute IOTaskStep peut demander un transfert de données à un ou plusieurs File(s), avec un volume total requis,
    #   un débit demandé, un type de transfert : READ ou WRITE, un type d'IO (séquentiel ou en plusieurs parties) et
    # L'ordonnanceur aura la possibilité de transférer un Fichier d'un niveau de stockage à un autre.
    # Chaque support de stockage a ses caractéristiques pour déterminer le temps des I/O, cf. mail de LM.
    # C'est le scheduler qui décide de quel fichier est placé à quel endroit.
    # La latence est une ordonnée à l'origine : un temps à attendre sans qu'une progression ne se fasse sur les I/O
    #   demandés.
    # On introduit un débit max en sortie de chaque nœud.
    # TODO La lecture du livre de Jalil amène à repenser les modèles énergétiques et temporels des dispositifs de
    #  stockage : le débit par I/O dépend fortement de la taille de chaque I/O (déjà implémenté via la latence), et
    #  la puissance et le temps dépendent, pour HDD et SSD, des écritures/lectures séquentielles/aléatoires.
    def __init__(self, name: str, capacity: int, throughput: float, latency: float, writing_malus: float = 1.):
        """
        Constructor of the Storage class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        :param writing_malus: float representing the ratio between the time to do a writing and the time to do a reading
        """
        self.name: str = name
        self.capacity: int = capacity
        self.throughput: float = throughput
        self.latency: float = latency
        self.files: Set["File"] = set()  # All the files on the Storage will be loaded later on.
        self.occupation: int = 0    # Number of bytes occupied on the Storage.
        self.running_tasks: List[Task] = []
        self.writing_malus = writing_malus  # TODO L'idée du writing_malus est à revoir dans le cadre des modifications
        #                                      générales à apporter aux modèles temporels et énergétiques des Storages.

    def __repr__(self):
        return f"Storage '{self.name}' " \
               f"with {pretty_print(self.occupation, 'B')} out of {pretty_print(self.capacity, 'B')} occupied " \
               f"throughput = {pretty_print(self.get_current_throughput_per_task(), 'B/s')} " \
               f"in {pretty_print(self.throughput, 'B/s')} " \
               f"for Tasks: {list(map(lambda task: task.name, self.running_tasks))} " \
               f"latency = {pretty_print(self.latency, 's')} "

    @abstractmethod
    def power_consumption(self):
        """
        Computes the current power consumption (W) of this Storage.
        This is only the skeleton of the methodNeeds to be implemented in each subclass.
        :return: A NotImplementedError
        """
        raise NotImplementedError("This is the method from class Storage, Call a method from a subclass.")

    def register(self, task: Task):
        """
        Give a Task part of the available I/O bandwidth.
        :param task: Newly elected Task for an I/O booking.
        :return: None
        """
        assert type(task.steps[task.current_step_index]).__name__ == "IOTaskStep"
        self.running_tasks.append(task)

    def unregister(self, task: Task):
        """
        Deallocate the I/O bandwidth that a Task has.
        :param task: The Task that liberates resources.
        :return: None
        """
        assert type(task.steps[task.current_step_index]).__name__ == "IOTaskStep"
        self.running_tasks.remove(task)

    def get_current_throughput_per_task(self, rw: str = None):
        """
        Get the bandwidth (B/s) allocated to one Task registered on the disk. The bandwidth is allocated uniformly.
        :param rw: a string stating if the I/O performed is Read or Write (useful for SSDs).
        :return: the amount of bandwidth available for one task (reading, or writing) on the disk.
        """
        return self.throughput / max(1, len(self.running_tasks))
