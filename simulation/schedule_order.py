from enum import auto, Enum
from typing import List, Tuple

from tools.utils import pretty_print
from tools.tasks.task import Task
from tools.node import Node
from tools.storage.abstract_storage_tier import Storage
from tools.storage.file import File


class Order(Enum):
    """
    Defines the different kinds of ScheduleOrders.
    """
    START_TASK = auto()
    START_IOTASKSTEP = auto()
    TRANSFER_FILE = auto()


class ScheduleOrder:
    """
    Class representing a decision of the scheduler that the Simulation shall execute at due time.
    Such a decision could be :
     - To allocate Nodes to a Task.
     - To allocate Storage's throughput to a TaskStep.
     - To execute a File transfert from a Storage to another.
    """

    def __init__(self, order: Order, time: float, task: Task = None, nodes: List[Tuple[Node, int]] = None,
                 file: File = None, dest_storage: Storage = None):
        """
        Constructor of the ScheduleOrder class.
        :param order: The kind of order to execute.
        :param time: The moment to execute the order. # TODO If time is 0. the simulation will translate it as 'now'.
        """
        self.order = order
        self.time: float = time
        self.task: Task = task  # If the order is to execute a new Task, or to execute an IOTaskStep.
        self.nodes: List[Tuple[Node, int]] = nodes  # If the order is to execute a new Task,
        # this list should hold tuples containing one Node and the number of cores to take on it.
        self.file: File = file  # If the order is to execute an IOTaskStep, or to transfer a File.
        self.dest_storage: Storage = dest_storage  # If the order is to transfer a File.

    def __repr__(self):
        r = ""
        if self.order:
            r += f"ScheduleOrder {self.order} "
        if self.time:
            r += f"at {pretty_print(self.time, 's')} "
        if self.task:
            r += f"for {self.task} "
        if self.nodes:
            r += f"on {self.nodes} "
        if self.file:
            r += f"on File {self.file} "
        if self.dest_storage:
            r += f"to go to Storage {self.dest_storage} "
        return r
