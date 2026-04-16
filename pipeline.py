"""
Pipeline for speaker diarization task.

This module contains the main pipeline for the speaker diarization task, 
including the live system and the embedding extraction process.
"""

import json
import logging
import numpy as np
import sounddevice as sd
from silero_vad import (load_silero_vad, 
                        VADIterator, 
                        read_audio, 
                        get_speech_timestamps)
import torch
import os
import argparse 
import torchaudio
import tqdm
from lib.const import *

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: [""]  

from lib.log import get_system_logger
from lib.vad_data_loader import VADDataLoader
from speechbrain.inference.classifiers import EncoderClassifier
from dotenv import load_dotenv
from pyannote.audio import Inference, Model
from lib.utils.xml_to_ref import load_refs_from_audio_file 
load_dotenv()

import simpleder

# load pretrained model
model = Model.from_pretrained("pyannote/embedding",
    use_auth_token=os.environ.get("HF_TOKEN")
)

# wrap it for inference
inference = Inference(model)



from lib.const import *

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

from lib.modules.vad.silero import SileroVADModule
from lib.modules.embeddings.pyannote_inference import PyannoteInferenceModule
from lib.modules.embeddings.speechbrain_inference import SpeechBrainInferenceModule
from lib.modules.helpers.vad_to_wav_segments import VADToWavSegmentsLoader 
from lib.modules.helpers.labels_to_hyp import LabelsToHypModule
from lib.modules.clustering.dbscan_clustering import DBSCANClusteringModule
from lib.modules.clustering.cspace_clustering import CSpaceClusteringModule
from lib.modules.visualization.embedding_visu import EmbeddingVisualizationModule
from lib.modules.overlap.pyannote_overlap_module import PyannoteOverlapModule 


class InitialDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = DBSCANClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.7)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der


class CSpaceDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        embeddings = SpeechBrainInferenceModule(wav_segments).execute()
        labels, data_plot = CSpaceClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        print(refs)
        print(hyp)

        self.der = simpleder.DER(refs, hyp, collar=0.7)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der

class WithOverlapDetectionDiarizationPipeline:
    def __init__(self):
        pass

    def run(self, audio_path: str, visualize: bool = False):
        logger.info("Dataset diarization system started.")

        vad_segments = SileroVADModule().execute(audio_path)
        wav_segments = VADToWavSegmentsLoader(audio_path, vad_segments).execute()
        overlaps = PyannoteOverlapModule(wav_segments).execute()
        logger.info(f"Detected {len(overlaps)} overlapping segments.")
        embeddings = PyannoteInferenceModule(wav_segments).execute()
        labels, data_plot = DBSCANClusteringModule(embeddings).execute()
        refs = load_refs_from_audio_file(audio_path)
        hyp = LabelsToHypModule(labels, vad_segments).execute()

        self.der = simpleder.DER(refs, hyp, collar=0.7)

        if visualize:
            EmbeddingVisualizationModule(labels, data_plot, wav_segments).execute()
        logger.info("Dataset diarization system finished.")

    def get_der(self):        
        return self.der




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Diarization Pipeline")
    parser.add_argument("--audio_path", type=str, default=TEST_AUDIO_PATH, help="Path to the input audio file.")
    parser.add_argument("--visualize", action="store_true", help="Show interactive embedding visualization.")
    args = parser.parse_args()

    #pipeline = InitialDiarizationPipeline()
    #pipeline.run(args.audio_path, visualize=args.visualize)

    #cspace_pipeline = CSpaceDiarizationPipeline()
    #cspace_pipeline.run(args.audio_path, visualize=args.visualize)

    overlap_aware_pipeline = WithOverlapDetectionDiarizationPipeline()
    overlap_aware_pipeline.run(args.audio_path, visualize=args.visualize)

    #logger.info(f"DER for DBSCAN Clustering: {pipeline.get_der()}")
    #logger.info(f"DER for C-Space Clustering: {cspace_pipeline.get_der()}")
