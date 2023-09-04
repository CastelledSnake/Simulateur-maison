from model.tasks.abstract_task_step import TaskStep
from enum import Enum, auto

from model.storage.hdd_storage_tier import HDD
from model.storage.ssd_storage_tier import SSD
from model.tasks.task import State


class IOPattern(Enum):
    """
    Describe an event.
    """

    SEQUENTIAL = auto()   # Sequential: faster on HDD
    UNKNOWN = auto()    # Random: slower on HDD


class IOTaskStep(TaskStep):
    """
    Describes a step of a task that is dedicated to IO.
    """

    def __init__(self, file: "File", rw: str, total_io_volume: int, average_io_size: int,
                 io_pattern: IOPattern = IOPattern.SEQUENTIAL):
        """
        Constructor of IOTaskStep class.
        :param total_io_volume: The total amount of IO required completing the step (B)
        """
        TaskStep.__init__(self)
        self.file: "File" = file

        self.rw = rw
        self.total_io_volume: int = total_io_volume
        self.average_io_size = average_io_size
        self.io_pattern = io_pattern

        self.available_bandwidth: float = 0.  # The amount of throughput available for I/OTS to perform IOs.
        self.progress: int = 0  # The number of bytes exchanged so far by the IOTaskStep.
        self.last_progress_increment: float = 0.  # The time at which the last evaluation of I/OTS progression occurred.

    def on_start(self, current_time: float):
        """
        Start the ComputeTaskStep, activate calculation on reserved cores.
        :param current_time: The current time at which the TaskStep effectively starts.
        :return: None
        """

        # assert that the step has not been started once before & set starting time
        assert self.progress == 0 and self.last_progress_increment == 0
        self.last_progress_increment = current_time

        # Throughput is a shared resource, that we consider here evenly shared between all tasks
        # We inform the used storage that we will be asking for a share for a while
        assert self.file.storage is not None
        self.file.storage.register(self.task)

        # assert that the task is under the proper state & update state
        assert self.task.state is State.EXECUTING
        self.task.state = State.EXECUTING_IO

    def on_finish(self):
        """
        End the ComputeTaskStep, deactivate calculation on reserved cores.
        :return: None
        """
        # We informa the used storage that we no longer need a share of the throughput
        assert self.file.storage is not None
        self.file.storage.unregister(self.task)

        # assert that the task is under the proper state & update state
        assert self.task.state is State.EXECUTING_IO
        self.task.state = State.EXECUTING

    def predict_finish_time(self, current_time: float):
        """
        Computes an estimation of the remaining time to complete the IOTaskStep,
        considering resources allocated and assuming there are no perturbations incoming in the system.
        :param current_time: The new moment isolated by the simulation.
        :return: The estimated remaining time in seconds (float)
        """
        assert self.file.storage is not None
        self.increment_progress(current_time)   # Updates progress done since the last evaluation,
        # with the last bandwidth available.
        remaining_volume = self.total_io_volume - self.progress
        self.available_bandwidth = self.file.storage.get_current_throughput_per_task(self.rw)

        if type(self.file.storage) is HDD:
            # if the storage type is HDD, sequential access is faster since latency is negligible
            if self.io_pattern is IOPattern.SEQUENTIAL:
                return remaining_volume / self.available_bandwidth
            else:
                return remaining_volume / self.available_bandwidth \
                    + (remaining_volume / self.average_io_size) * self.file.storage.latency
        elif type(self.file.storage) is SSD:
            # TODO Actuellement, la différence de temps entre L et E dans un SSD est donnée par un malus de 20% pour E.
            eta = remaining_volume / self.available_bandwidth \
                + (remaining_volume / self.average_io_size) * self.file.storage.latency
            if self.rw == "r":
                return eta
            elif self.rw == "w":
                return self.file.storage.writing_malus * eta
            else:
                raise ValueError(f"Unrecognized kind of I/O : {self.rw}")
        else:
            raise RuntimeError(f'Using unsupported file storage class {type(self.file.storage)}')

    def increment_progress(self, current_time: float):
        """
        Computes the current progression of the IOTaskStep.
        :param current_time: The new moment isolated by the simulation.
        :return: None
        """
        assert current_time >= self.last_progress_increment
        self.progress += int(
            (current_time - self.last_progress_increment) * self.available_bandwidth / self.file.storage.writing_malus)
        assert self.progress >= 0
        self.last_progress_increment = current_time
