from qgym.generators import BasicInteractionGenerator


class CurriculumScheduler:
    def __init__(self, start_length, max_length, proceed_threshold, retreat_threshold, curriculum_generator):
        self.start_length = start_length
        self.max_length = max_length
        self.proceed_threshold = proceed_threshold
        self.retreat_threshold = retreat_threshold
        self.curriculum_generator = curriculum_generator

        self.current_length = start_length

    def _make_schedule(self):
        return self.curriculum_generator.generate(self.current_length)

    def update_schedule(self, performance):
        if performance >= self.retreat_threshold:
            self.current_length -= 1
        elif performance <= self.proceed_threshold:
            self.current_length += 1

        return self._make_schedule()

class GateLengthCurriculum:
    def generate(self, level):
        return BasicInteractionGenerator(max_length=level)

def compute_performance(swaps, baseline_swaps):
    return swaps - baseline_swaps