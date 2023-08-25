from enum import auto, Enum
from model.storage.abstract_storage_tier import Storage
from model.utils import pretty_print


class SSDState(Enum):
    """
    Describes the possible states for SSDs.
    """

    ACTIVE = auto()
    SLEEP = auto()


class SSD(Storage):
    """
    Class describing a storage device of type Solid-State Drive: a set of flash memory units that can keep data.
    """

    def __init__(self, name: str, capacity: int, throughput: float, latency: float, max_read_power: float,
                 max_write_power: float, leakage_power: float, sleep_power: float = 0.):
        """
        Constructor of the SSD class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        :param max_read_power: Power consumed if all the bandwidth is used to read.
        :param max_write_power: Power consumed if all the bandwidth is used to write.
        :param leakage_power: Power consumed by the disk for its own needs.
        :param sleep_power: Power consumed by the SSD when sleeping (default 0)
        """
        super().__init__(name, capacity, throughput, latency)
        self.max_read_power: float = max_read_power
        self.max_write_power: float = max_write_power
        self.leakage_power: float = leakage_power
        self.sleep_power: float = sleep_power
        self.flow_read: float = 0.    # Amount of bandwidth (B/s) taken to read data.
        self.flow_write: float = 0.   # Amount of bandwidth (B/s) taken to write data.
        self.state = SSDState.ACTIVE

    def __repr__(self):
        return f"SSD '{self.name}' " \
               f"with {pretty_print(self.occupation, 'B')} out of {pretty_print(self.capacity, 'B')} occupied " \
               f"global throughput = {pretty_print(self.get_current_throughput_per_task(), 'B/s')} in {pretty_print(self.throughput, 'B/s')} " \
               f"of which read = {pretty_print(self.flow_read, 'B/s')} " \
               f"and write = {pretty_print(self.flow_write, 'B/s')} " \
               f"for Tasks: {list(map(lambda task: task.name, self.running_tasks))} " \
               f"currently {self.state} "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this SSD.
        :return: The power consumption (W) as a float.
        This formula comes from this study :
        """
        # Proposition pour SSD :
        # Params : les données constructeurs self.max_write_power, self.max_read_power, self.idle_power
        #   Distinguer 2 modes :
        #       Actif : Le disque normal, P = (flow_w/BP)*mwp + (flow_r/BP)*mrp + idle_power
        #       Veille : Le disque ne peut pas lire/écrire, sa consommation vaut self.idle_power
        if self.state == SSDState.ACTIVE:
            return (self.flow_read*self.max_read_power + self.flow_write*self.max_write_power)\
                / self.throughput + self.leakage_power
        elif self.state == SSDState.SLEEP:
            return self.sleep_power
        else:
            raise ValueError(f"SSD {self} has state {self.state}.")