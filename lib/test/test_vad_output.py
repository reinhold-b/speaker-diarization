"""
Here i want to test the vad output by playing the chunks calculated.
"""

from lib.vad_data_loader import VADDataLoader
from lib.const import *
import sounddevice as sd 
import numpy as np
import time

def play_segments(segments, sample_rate: int):
    for i, seg in enumerate(segments):
        # falls seg schon ein ndarray ist
        audio = seg

        # optional: falls dein Loader dicts zurückgibt
        if isinstance(seg, dict) and "waveform" in seg:
            audio = seg["waveform"]

        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.size == 0:
            continue

        # optionales Normalisieren (nur wenn nötig)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        print(f"Playing segment {i + 1}/{len(segments)} ({len(audio)/sample_rate:.2f}s)")
        sd.play(audio, samplerate=sample_rate, blocking=True)
        time.sleep(0.05) 


def main():
    data = VADDataLoader.load_from_json(
        filepath_vad=TEST_VAD_PATH,
        path_audio=TEST_AUDIO_PATH,
    )

    play_segments(data, sample_rate=16000)


if __name__ == "__main__":
    main()