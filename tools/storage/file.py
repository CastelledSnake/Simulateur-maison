from tools.storage.abstract_storage_tier import Storage
from tools.utils import pretty_print


class File:
    """
    Class describing a file on the system. These files can be red or written by IOTaskSteps, allowing to tools I/Os.
    """

    def __init__(self, name: str, size: int, preferred_storage: Storage, storage: Storage = None):
        """
        Constructor of the File class.
        :param name: The name of the File instance.
        :param size: The number of bytes the File occupies (this number is mutable).
        :param preferred_storage: The preferred Storage device to put the File on.
        :param storage: The Storage device the File is written on.
        """
        self.name: str = name
        self.size: int = size
        self.preferred_storage: Storage = preferred_storage
        self.storage: Storage = storage

    def __str__(self):
        return f"File '{self.name}', taking {pretty_print(self.size, 'B', 3)} " \
               f"on {self.storage.name}, preferred : {self.preferred_storage.name}"


"""
Pour implémenter le transfert de fichiers d'un disque à l'autre :
    - Dans la classe Storage, créer un ensemble de pseudo-tâches nommées 'transferts' qui viennent s'ajouter en
    concurrence avec les tâches normales pour la réalisation d'I/O.
        Ces transferts se font d'un disque vers un autre ==> Ils occupent de la BP sur les 2 disques.
            Plus précisément : min(BP_disponible_disque_1, BP_disponible_disque_2)
        La méthode get_current_throughput_per_task() en est impactée sur les 2 disques.
    - Dans la classe File, créer une fonction qui entame un transfert.
    - Dans la classe simulation, introduire la gestion de ces événements (une fois les ScheduleOrder réintroduits):
        file_transfert_begin.
        file_transfert_end.
"""
