from enum import auto, Enum
from tools.storage.abstract_storage_tier import Storage
from tools.utils import pretty_print


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
                 max_write_power: float, leakage_power: float, sleep_power: float = 0., writing_malus: float = 1.2):
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
        :param writing_malus: float representing the ratio between the time to do a writing and the time to do a reading
        """
        super().__init__(name, capacity, throughput, latency, writing_malus)
        self.max_read_power: float = max_read_power
        self.max_write_power: float = max_write_power
        self.leakage_power: float = leakage_power
        self.sleep_power: float = sleep_power
        self.state = SSDState.ACTIVE

    def __repr__(self):
        flows = self.get_flows_read_and_write()
        return f"SSD '{self.name}' " \
               f"with {pretty_print(self.occupation, 'B')} out of {pretty_print(self.capacity, 'B')} occupied " \
               f"global throughput = {pretty_print(self.get_current_throughput_per_task('r'), 'B/s')} in " \
               f"{pretty_print(self.throughput, 'B/s')} " \
               f"of which read = {pretty_print(flows['flow_read'], 'B/s')} " \
               f"and write = {pretty_print(flows['flow_write'], 'B/s')} " \
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
        #       Actif : Le disque normal, P = (flow_w/BP)*mwp + (flow_r/BP)*mrp + self.leakage_power
        #       Veille : Le disque ne peut pas lire/écrire, sa consommation vaut self.sleep_power
        if self.state == SSDState.ACTIVE:
            flows = self.get_flows_read_and_write()
            return (flows["flow_read"] * self.max_read_power + flows["flow_write"] * self.max_write_power) \
                / self.throughput + self.leakage_power
        elif self.state == SSDState.SLEEP:
            return self.sleep_power
        else:
            raise ValueError(f"SSD {self} has state {self.state}.")

    def get_flows_read_and_write(self):
        """
        Get the amounts of bandwidth (B/s) allocated respectively to reading and writing I/Os.
        :return: a float
        """
        if len(self.running_tasks) == 0:
            return {"flow_read": 0., "flow_write": 0.}
        else:
            count_read = 0
            count_write = 0
            for task in self.running_tasks:
                if task.steps[task.current_step_index].rw == "r":
                    count_read += 1
                else:
                    count_write += 1
            return {"flow_read": self.throughput * count_read / len(self.running_tasks),
                    "flow_write": self.throughput * count_write / len(self.running_tasks)}

    def register(self, task: "Task"):
        super().register(task)

    def unregister(self, task: "Task"):
        super().unregister(task)

    def get_current_throughput_per_task(self, rw: str = None):
        """
        Get the bandwidth (B/s) allocated to one Task registered on the disk. The bandwidth is allocated uniformly.
        :param rw: a string stating if the I/O performed is Read or Write (useful for SSDs).
        :return: the amount of bandwidth available for one task (reading, or writing) on the disk.
        """
        if rw == "r":
            return self.throughput / max(1, len(self.running_tasks))
        elif rw == "w":
            return self.throughput / max(1, len(self.running_tasks)) * self.writing_malus
        else:
            raise ValueError(f"Unrecognised kind of IO : {rw}")
