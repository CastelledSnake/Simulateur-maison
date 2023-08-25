from model.storage.abstract_storage_tier import Storage
from model.utils import pretty_print


class File:
    """
    Class describing a file on the system. These files can be red or written by IOTaskSteps, allowing to model I/Os.
    """

    def __init__(self, name: str, size: int, preferred_storage: Storage, storage: Storage = None):
        """
        Constructor of the File class.
        :param name: The name of the File instance.
        :param size: The number of bytes the File occupies (this number is mutable).
        :param preferred_storage: The preferred kind of Storage device to put the File on.
        """
        self.name: str = name
        self.size: int = size
        self.preferred_storage: Storage = preferred_storage
        self.storage: Storage = storage

    def __str__(self):
        return f"File '{self.name}', taking {pretty_print(self.size, 'B', 3)} " \
               f"on {self.storage}, preferred : {self.preferred_storage}"
