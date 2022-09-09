import sys
from os.path import dirname

sys.path.append(dirname(dirname(__file__)))

from mel2wav import MelVocoder

from pathlib import Path
import argparse
import librosa
import torch

load_path = "/home/evardanyan/dev/melgan-neurips/results/lj_ngf16_spec0.4"
wav_path = "/home/evardanyan/dev/melgan-neurips/results/lj_ngf16_spec0.4/original_3.wav"


def main():
    vocoder = MelVocoder(load_path, device="cuda")

    wav, sr = librosa.core.load(wav_path)

    mel= vocoder(torch.from_numpy(wav)[None])
    full_wave = vocoder.inverse(mel).squeeze().cpu().numpy()
    part_wave = vocoder.inverse(mel[...,:-50]).squeeze().cpu().numpy()
    
    print(full_wave[...,:part_wave.shape[-1]] / part_wave)



if __name__ == "__main__":
    main()
