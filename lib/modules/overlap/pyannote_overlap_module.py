import os

import numpy as np
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

from lib.models.module import DiarizationModule
from lib.const import SAMPLING_RATE

load_dotenv()


class PyannoteOverlapModule(DiarizationModule):
    """Return indices of WAV segments that contain overlapped speech."""

    def __init__(
        self,
        wav_segments: list[np.ndarray],
        threshold: float = 0.5,
        model_name: str = "pyannote/overlapped-speech-detection",
    ):
        super().__init__(tag="Pyannote Overlap Detection")
        self.wav_segments = wav_segments
        self.threshold = threshold
        self.model_name = model_name
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            token=os.getenv("HF_TOKEN"),
        )

    def _run_on_segment(self, segment: np.ndarray) -> bool:
        if segment is None or len(segment) == 0:
            return False

        waveform = torch.tensor(np.asarray(segment), dtype=torch.float32).unsqueeze(0)
        annotation = self.pipeline({"waveform": waveform, "sample_rate": SAMPLING_RATE})

        return any(True for _ in annotation.itertracks(yield_label=True))

    def run(self):
        overlap_indices = []

        for idx, segment in enumerate(self.wav_segments):
            if self._run_on_segment(segment):
                overlap_indices.append(idx)

        return overlap_indices