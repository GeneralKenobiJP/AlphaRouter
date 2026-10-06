from qgym.generators import BasicInteractionGenerator
from qgym.generators.interaction import InteractionGenerator


class CurriculumScheduler:
    """
    Implements curriculum learning.
    The idea is to start with simpler examples, and then gradually progress to
    more difficult ones as the agent learns.

    The scheduler itself controls the process by specifying the curriculum generator, start and max levels of the curriculum,
    and the performance thresholds for advancing or backtracking with the curriculum.
    """
    def __init__(self, start_level=1, max_level=10, inc_value=1, proceed_threshold=1.5, retreat_threshold=3.0,
                 cooldown = 10, curriculum_generator = None):
        """
        Parameters
        ----------
        start_level - the starting (minimum) level of the curriculum
        max_level - the maximum level of the curriculum
        inc_value - specifies by how many levels we advance each time we proceed further with the curriculum.
        proceed_threshold - performance threshold for increasing the difficulty of the curriculum.
            The definition of performance can be freely adjusted, but it is assumed to be our_swaps / baseline_swaps.
            (NOTE: the smaller number, the better)
        retreat_threshold - performance threshold for decreasing the difficulty of the curriculum.
            The definition of performance can be freely adjusted, but it is assumed to be our_swaps / baseline_swaps.
            (NOTE: the smaller number, the better)
        cooldown - after changing the level of curriculum, we wait for the specified amount of episodes until
            the level can be changed again, so that the training has the chance to stabilise with the new curriculum.
        curriculum_generator - the generator of the curriculum. E.g. level = number of gates.
        """
        self.start_level = start_level
        self.max_level = max_level
        self.inc_value = inc_value
        self.proceed_threshold = proceed_threshold
        self.retreat_threshold = retreat_threshold
        if curriculum_generator is None:
            curriculum_generator = GateLengthCurriculumGenerator()
        self.curriculum_generator = curriculum_generator
        self.cooldown = cooldown

        self.current_level = start_level
        self.counter = 0

        self.level_log = []

    def make_schedule(self) -> InteractionGenerator:
        """
        Logs the current level of the curriculum, and generates a new schedule, given the current level.
        Returns
        -------
        InteractionGenerator that generates circuits under the new curriculum.
        """
        self.level_log.append(self.current_level)
        return self.curriculum_generator.generate(self.current_level)


    def update_schedule(self, performance, force_increase = False) -> bool:
        """
        Checks the current performance against the thresholds and updates the curriculum accordingly.
        If the scheduler is in the cooldown period after the last curriculum change, the curriculum remains the same.

        Parameters
        ----------
        performance - The current performance of our routing agent.
            The definition of performance can be freely adjusted, but it is assumed to be our_swaps / baseline_swaps.
            (NOTE: the smaller number, the better)
        force_increase - if True, the curriculum will always advance iff there is no cooldown period at the moment.

        Returns
        -------
        True if the curriculum was updated, False otherwise.
        """
        self.level_log.append(self.current_level)
        update = False

        self.counter += 1

        # print(f"Counter: {self.counter}")

        if self.counter < self.cooldown:
            return False

        if force_increase or (self.current_level < self.max_level and performance <= self.proceed_threshold):
            self.counter = 0
            update = True
            self.current_level += self.inc_value
            print("Curriculum proceeding")
        elif self.current_level > self.start_level and performance >= self.retreat_threshold:
            self.counter = 0
            update = True
            self.current_level -= self.inc_value
            print("Curriculum retreat")

        print(f"Current level: {self.current_level}")

        return update

class GateLengthCurriculumGenerator:
    """
    The most basic curriculum generator. Level = the number of gates in the circuit.
    """
    def generate(self, level) -> InteractionGenerator:
        return BasicInteractionGenerator(max_length=level)

def compute_performance(swaps, baseline_swaps) -> float:
    """
    Computes the performance of the routing agent, according to the following definition:
        our swaps / baseline_swaps
    If baseline_swaps == 0 (i.e. no routing is needed), the output is swaps + 1.
    Parameters
    ----------
    swaps - the number of SWAP gates inserted by our agent.
    baseline_swaps - the number of SWAP gates inserted by the baseline router.
    """
    if baseline_swaps == 0:
        print(f"performance: {swaps + 1}")
        return swaps + 1
    print(f"performance: {swaps / baseline_swaps}")
    return swaps / baseline_swaps