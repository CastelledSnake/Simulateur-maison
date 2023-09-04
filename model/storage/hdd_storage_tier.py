from enum import auto, Enum

from model.storage.abstract_storage_tier import Storage
from model.utils import pretty_print


class HDDState(Enum):
    """
    Class describing all the states an HDD can take.
    """

    NOMINAL = auto()
    SLOW = auto()
    SLEEP = auto()


class HDD(Storage):
    """
    Class describing a storage device of type Hard Disk Drive: a hard disk always spinning and an arm moving on top of
    the disk to perform reads and writes (as well as the embedded electronics).
    """

    def __init__(self, name: str, capacity: int, throughput: float, latency: float,
                 disk_radius: float, disk_max_spin: float, disk_min_spin: float,
                 arm_mass: float, arm_radius: float, arm_max_spin: float,
                 electronic_power: float, sleep_power: float = 0., writing_malus = 1.):
        """
        Constructor of the HDD class.
        :param name: The name of the storage instance (str)
        :param capacity: Number of bytes available on the device.
        :param throughput: I/O-bandwidth (B/s) to communicate with the outside.
        :param latency: Time (s) required communicating any piece of data with any other device of the simulation.
        :param disk_radius: Radius (m) of the spinning disk.
        :param disk_max_spin: Maximum angular speed (rad/s) allowed for the disk ().
        :param disk_min_spin: Minimum angular speed (rad/s) allowed for the disk (taken in idle mode).
        :param arm_mass: Mass (kg) of the I/O arm
        :param arm_radius: Radius (m) of the I/O arm
        :param electronic_power: Power consumed by all electronic components of the HDD
        :param sleep_power: power consumed by the HDD when sleeping (default 0)
        :param writing_malus: float representing the ratio between the time to do a writing and the time to do a reading
        """
        super().__init__(name, capacity, throughput, latency)
        self.disk_radius: float = disk_radius
        self.disk_max_spin: float = disk_max_spin
        self.disk_min_spin: float = disk_min_spin
        self.arm_momentum: float = arm_mass * (2*arm_radius)**2 / 12
        self.arm_radius: float = arm_radius
        self.arm_max_spin: float = arm_max_spin
        self.electronic_power: float = electronic_power
        self.sleep_power = sleep_power
        self.disk_current_spin: float = disk_max_spin  # Let suppose that the HDD begins in nominal mode.
        self.arm_current_spin: float = arm_max_spin    # Let suppose that the HDD begins in nominal mode.
        self.state = HDDState.NOMINAL

    def __repr__(self):
        return f"HDD '{self.name}' " \
               f"with {pretty_print(self.occupation, 'B')} out of {pretty_print(self.capacity, 'B')} occupied " \
               f"throughput = {pretty_print(self.get_current_throughput_per_task(), 'B/s')} " \
               f"in {pretty_print(self.throughput, 'B/s')} " \
               f"for Tasks: {list(map(lambda task: task.name, self.running_tasks))} " \
               f"currently {self.state} with disk_spin = {pretty_print(self.disk_current_spin, 'rad/s')} " \
               f"and arm_speed = {pretty_print(self.arm_current_spin, 'rad/s')} currently {self.state} "

    def power_consumption(self):
        """
        Computes the current power consumption (W) of this HDD.
        :return: The power consumption (W) as a float.
        This formula comes from this study :
        """
        # Proposition pour HDD :
        #   Réutiliser la formule de l'étude de 1990 : P_SPM = w_SPM**2.8 * (2r)**4.6 (on prend 1 unique disque par HDD)
        #   TODO analyse dimensionnelle.
        #    Les formules données donnent des résultats aberrants de plusieurs ordres de grandeur.
        #   Pour le VCM : ajouter une formule inspirée d'une autre étude : E_VCM = w_VCM**2 * J_VCM / 2
        #   (cf. cours pour calculer le moment d'inertie)
        #   Pour l'électronique, ajouter un petit offset au total.
        #   Introduire la notion de "repos" : le bras ne tourne plus,
        #   et la "veille" : le disque tourne au ralenti et le bras est immobile.
        if self.state == HDDState.SLEEP:
            return self.sleep_power
        elif self.state == HDDState.SLOW:
            return self.disk_current_spin**2.8 * (2 * self.disk_radius)**4.6 + self.electronic_power
        elif self.state == HDDState.NOMINAL:
            # TODO les coefficients numériques en début des 2 lignes sont hors du modèle.
            p_spm = 1e-9 * self.disk_current_spin**2.8 * (2 * self.disk_radius)**4.6
            p_vcm = 1e8 * self.arm_current_spin ** 2 * self.arm_momentum / 2
            return p_spm + p_vcm + self.electronic_power
        else:
            raise ValueError(f"SSD {self} has state {self.state}.")
