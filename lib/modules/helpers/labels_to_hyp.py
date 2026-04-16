from typing import Iterable

from lib.models.module import DiarizationModule
from lib.const import SAMPLING_RATE


class LabelsToHypModule(DiarizationModule):
    """
    Build hypothesis tuples in the same structure as xml_to_ref:
    [(speaker_tag, start, end), ...]
    """

    def __init__(
        self,
        labels: Iterable[int],
        data_points: list[dict],
        sample_rate: int = SAMPLING_RATE,
        timestamps_in_seconds: bool = False,
    ):
        super().__init__(tag="Labels to HYP")
        self.labels = list(labels)
        self.data_points = data_points
        self.sample_rate = sample_rate
        self.timestamps_in_seconds = timestamps_in_seconds

    def _to_seconds(self, value: float) -> float:
        if self.timestamps_in_seconds:
            return float(value)
        return float(value) / float(self.sample_rate)

    def run(self):
        hyp = []

        n = min(len(self.labels), len(self.data_points))
        for i in range(n):
            point = self.data_points[i]
            if not isinstance(point, dict):
                continue
            if "start" not in point or "end" not in point:
                continue

            start_sec = self._to_seconds(float(point["start"]))
            end_sec = self._to_seconds(float(point["end"]))

            if end_sec <= start_sec:
                continue

            label = int(self.labels[i])

            if label < 0:
                continue

            speaker_tag = f"spk_{label}" if label >= 0 else "spk_noise"
            hyp.append((speaker_tag, float(start_sec), float(end_sec)))

        hyp.sort(key=lambda x: float(x[1]))
        return hyp
