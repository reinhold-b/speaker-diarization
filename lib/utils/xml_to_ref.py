"""
Convert XML reference files
to python objects to use for DER calculation.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)


def xml_to_ref(filename: str):
    base_dir = Path(__file__).resolve().parents[2]
    file_path = base_dir / filename
    speaker_tag = filename.split("/")[-1].split(".")[1]
    tree = ET.parse(file_path)
    root = tree.getroot()

    ref = []
    for segment in root.findall("segment"):
        start = segment.attrib.get("transcriber_start")
        end = segment.attrib.get("transcriber_end")
        if start is not None and end is not None:
            ref.append((speaker_tag, float(start), float(end)))

    return ref

def load_refs_from_audio_file(audio_file: str):
    base_dir = Path(__file__).resolve().parents[2]
    print(base_dir)
    file_path = base_dir.joinpath("datasets/amicorpus/ami_public_manual_1.6.2/segments")
    keyword = audio_file.split("/")[-1].split(".")[0]
    print(file_path)
    print(keyword)
    refs = []
    for file in file_path.rglob(f"*{keyword}*"):
        logger.info(f"Processing file: {file}")
        ref = xml_to_ref(str(file))
        refs.extend(ref)

    refs.sort(key=lambda x: float(x[1])) 
    return refs

if __name__ == "__main__":

    path = "/Users/reinholdb_/Desktop/ITA/sem6/slp/project/speaker-diarization/datasets/amicorpus/ami_public_manual_1.6.2/segments"
    wpath = Path(path)
    keyword = "ES2016a"
    refs = []
    for file in wpath.rglob(f"*{keyword}*"):
        logger.info(f"Processing file: {file}")
        ref = xml_to_ref(str(file))
        refs.extend(ref)


    refs.sort(key=lambda x: float(x[1])) 

    logger.info(f"Processed {len(refs)} segments for keyword '{keyword}'.")
