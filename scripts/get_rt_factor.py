import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

from mel2wav import MelVocoder

import time
from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.set_num_threads(1)
    vocoder = MelVocoder(args.load_path, device="cpu")

    overall_duration = 0.0
    overall_time = 0.0
    for i, fname in tqdm(enumerate(args.folder.glob("*.wav"))):
        wav, sr = librosa.core.load(fname)

        mel= vocoder(torch.from_numpy(wav)[None])
        start = time.time()
        vocoder.inverse(mel).squeeze().cpu().numpy()
        overall_time += time.time() - start
        overall_duration += wav.size / sr
    
    print(f"Overall Duration of Audios: {overall_duration}")
    print(f"Overall Time spent on processing: {overall_time}")
    print(f"Real Time Factor: {overall_time / overall_duration}")


if __name__ == "__main__":
    main()
