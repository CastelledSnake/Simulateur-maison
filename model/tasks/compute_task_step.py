from model.tasks.abstract_task_step import TaskStep

from model.tasks.task import State


class ComputeTaskStep(TaskStep):
    """
    Describes a step of a task that is dedicated to computation.
    For this class, progression is measured with an integer (the amount of flop done)
    """

    def __init__(self, flop: int):
        """
        Constructor of ComputeTaskStep class.
        :param flop: The number of floating point operations that requires the step to complete (int).
        """
        TaskStep.__init__(self)
        self.flop: int = flop  # TBD
        self.progress: int = 0  # The progression of the task's execution.
        self.last_progress_increment: float = 0.  # The time at which the last evaluation of TaskStep's progression occurred.

    def on_start(self, current_time: float):
        """
        Start the ComputeTaskStep, activate calculation on reserved cores.
        :param current_time: The current time at which the TaskStep effectively starts.
        :return: None
        """

        # assert that the step has not been started once before & set starting time
        assert self.progress == 0 and self.last_progress_increment == 0
        self.last_progress_increment = current_time

        # Compute resources are NOT given for a single compute task step, they are given for a whole task
        # However, we update the core state during the compute task steps
        for node, allocated_core_count in self.task.allocated_cores:
            node.idle_busy_cores -= allocated_core_count

        # assert that the task is under the proper state & update state
        assert self.task.state is State.EXECUTING
        self.task.state = State.EXECUTING_CALCULATION

    def on_finish(self):
        """
        End the ComputeTaskStep, deactivate calculation on reserved cores.
        :return: None
        """
        for node, allocated_core_count in self.task.allocated_cores:
            node.idle_busy_cores += allocated_core_count

        # assert that the task is under the proper state & update state
        assert self.task.state is State.EXECUTING_CALCULATION
        self.task.state = State.EXECUTING

    def predict_finish_time(self):
        """
        Computes an estimation of the remaining time to complete the ComputeTaskStep,
            considering resources allocated and assuming there are no perturbations incoming in the system.
        :return: The estimated remaining time in seconds (float)
        """
        # Compute the available flops as of now
        available_flops = 0
        for node, allocated_cores_count in self.task.allocated_cores:
            available_flops += node.frequency * allocated_cores_count

        # If the simulation class is doing its job, the task step should never overflow
        assert self.progress <= self.flop

        return (self.flop - self.progress) / available_flops

    def increment_progress(self, current_time: int):
        """
        Computes the current progression of the ComputeTaskStep.
        :return: None
        """
        # Compute the available flops as of now
        available_flops = 0
        for node, allocated_cores_count in self.task.allocated_cores:
            available_flops += node.frequency * allocated_cores_count

        self.progress += int((current_time - self.last_progress_increment) * available_flops)
        assert self.progress >= 0
        self.last_progress_increment = current_time