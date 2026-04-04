import logging
import numpy as np
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator
import torch
import os

import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    print("here")
    torchaudio.list_audio_backends = lambda: [""]  # type: ignore

from lib.log import get_system_logger

from speechbrain.inference.classifiers import EncoderClassifier

from dotenv import load_dotenv

from pyannote.audio import Inference, Model
load_dotenv()

# load pretrained model
model = Model.from_pretrained("pyannote/embedding",
    use_auth_token=os.environ.get("HF_TOKEN")
)

# wrap it for inference
inference = Inference(model)



SAMPLING_RATE = 16_000
VAD_WINDOW_SIZE = 512

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

logger = get_system_logger()
silero_vad_model = load_silero_vad()
silero_iterator = VADIterator(silero_vad_model, sampling_rate=SAMPLING_RATE)

#sb_classifier = EncoderClassifier.from_hparams(
#    source="speechbrain/spkrec-ecapa-voxceleb",
#    savedir="pretrained_models/ecapa",
#    token=os.environ.get("HF_TOKEN")
#)

def extract_embedding(speaker_audio):
    """
    Use the VAD audio chunks to create speaker embeddings.
    """
    if speaker_audio is None or len(speaker_audio) == 0:
        return None

    audio_tensor = torch.tensor(speaker_audio, dtype=torch.float32).unsqueeze(0)
    embedding = inference({
        "waveform": audio_tensor,
        "sample_rate": SAMPLING_RATE,
    })

    if torch.is_tensor(embedding):
        embedding = embedding.squeeze().detach().cpu().numpy()
    else:
        embedding = np.asarray(embedding).squeeze()

    logger.debug(embedding)

    with open("data.csv", "a+") as f:
        np.savetxt(f, embedding.reshape(1, -1), delimiter=",")

    return embedding

def main():
    logger.info("Diarization system started.")

    # 512 samples at 16 kHz = 32 ms latency per VAD decision
    chunk_size = VAD_WINDOW_SIZE
    audio_buffer = np.array([], dtype=np.float32)
    is_speaking = False
    current_speech_chunks = []

    def callback(indata, frames, time, status):
        nonlocal audio_buffer, is_speaking, current_speech_chunks

        if status:
            logger.warning(f"Audio callback status: {status}")

        audio_chunk = indata[:, 0].astype(np.float32)
        audio_buffer = np.concatenate((audio_buffer, audio_chunk))

        while len(audio_buffer) >= VAD_WINDOW_SIZE:
            chunk = audio_buffer[:VAD_WINDOW_SIZE]
            audio_buffer = audio_buffer[VAD_WINDOW_SIZE:]

            speech_dict = silero_iterator(chunk, return_seconds=True)

            if is_speaking:
                current_speech_chunks.append(chunk)

            if speech_dict and speech_dict.get("start") and not is_speaking:
                is_speaking = True
                current_speech_chunks = [chunk]
                logger.info("Speaking")
            elif speech_dict and speech_dict.get("end") and is_speaking:
                is_speaking = False
                logger.info("Stopped speaking.")
                speech_audio = np.concatenate(current_speech_chunks) if current_speech_chunks else np.array([], dtype=np.float32)
                extract_embedding(speech_audio)
                current_speech_chunks = []


    with sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        callback=callback,
        blocksize=chunk_size
    ):
        logger.info("Recording started. Speak now (Ctrl+C to stop)")
        while True:
            sd.sleep(50)


if __name__ == "__main__":
    logger.debug("Starting...")
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        silero_iterator.reset_states()
