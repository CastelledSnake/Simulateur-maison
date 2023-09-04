from enum import auto, Enum
from typing import Set


from model.utils import pretty_print
from model.tasks.task import Task
from simulation.schedule_order import ScheduleOrder


class Event(Enum):
    """
    Describe an event.
    """

    SIMULATION_START = auto()   # The Simulation just began.
    TASK_SUBMIT = auto()    # A Task is to be coming to Simulation's view.
    TASK_BEGIN = auto()     # A task is to be allocated resources for execution.
    CALC_STEP_BEGIN = auto()    # A Task is to begin a ComputeTaskStep
    IO_STEP_BEGIN = auto()  # A Task is to begin an IOTaskStep
    CALC_STEP_END = auto()  # A Task is to end a ComputeTaskStep
    IO_STEP_END = auto()    # A Task is to end an IOTaskStep
    TASK_END = auto()   # A Task is about to end (i.e. its last TaskStep is to end).
    FILE_MOVE_BEGIN = auto()    # A File is to be leaving a Storage.
    FILE_MOVE_END = auto()  # A File is to be moved to a new Storage.
    SIMULATION_TERMINAISON = auto()     # The Simulation is over, because of no remaining Task to execute.


class NextEvent:
    """
    Data class containing all information required to describe the next event of the Simulation.
    """

    def __init__(self, events: Set[Event], task: Task or None, time: float, order: ScheduleOrder = None):
        """
        Constructor of the NextEvent class.
        :param events: The set of Events to be happening.
        The set usually have only 1 element, but can have 2 when a Task ends.
        :param task: The Task relative to this element (can be None if 'event' is about Simulation begin/end).
        :param time: The time of event happening.
        :param order: The ScheduleOrder related to the event, if it has been planned. Optional parameter.
        """
        self.events: Set[Event] = events
        self.task: Task or None = task
        self.time: float = time
        self.order: ScheduleOrder = order

    def __repr__(self):
        r: str = f"NEXT_EVENT : {self.events} "
        if self.task:
            r += f"on task {self.task}[{self.task.current_step_index}] "
        r += f"at {pretty_print(self.time, 's')} "
        if self.order:
            r += f"{self.order}"
        return r
