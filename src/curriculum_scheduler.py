from qgym.generators import BasicInteractionGenerator
from qgym.generators.interaction import InteractionGenerator


class CurriculumScheduler:
    def __init__(self, start_level=1, max_level=10, inc_value=1, proceed_threshold=1.5, retreat_threshold=3.0,
                 cooldown = 10, curriculum_generator = None):
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
        self.level_log.append(self.current_level)
        return self.curriculum_generator.generate(self.current_level)

    # def set_level(self, level):
    #     self.current_level = level
    #     self.counter = 0
    #
    # def increase_level(self):
    #     self.current_level += 1
    #     self.counter = 0

    def update_schedule(self, performance, force_increase = False) -> bool:
        self.level_log.append(self.current_level)
        update = False

        self.counter += 1

        print(f"Counter: {self.counter}")

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
    def generate(self, level) -> InteractionGenerator:
        return BasicInteractionGenerator(max_length=level)

def compute_performance(swaps, baseline_swaps):
    if baseline_swaps == 0:
        print(f"performance: {swaps + 1}")
        return swaps + 1
    print(f"performance: {swaps / baseline_swaps}")
    return swaps / baseline_swaps