from typing import List
import scipy.io.wavfile
import torch
import torch.nn.functional as F


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def get_magnitude_melspec(wave: torch.Tensor, n_fft: int):
    """
    Calculates the magnitude mel spectrogram of the wave 
    with win_length=n_fft and hop_length=n_fft/4
    """
    window = torch.hann_window(800).float()
    stft = torch.stft(
        wave, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
        window=window, center=True, return_complex=False)
    mag_spec = torch.sqrt((stft ** 2).sum(-1))  # magnitude = sqrt(real**2 + imaginary**2)
    return mag_spec


def spec_reconstruction_loss(wave_1: torch.Tensor, wave_2: torch.Tensor, n_ffts: List[int]):
    loss = 0
    for n_fft in n_ffts:
        spec_1 = get_magnitude_melspec(wave_1, n_fft)
        spec_2 = get_magnitude_melspec(wave_2, n_fft)
        loss = loss + F.l1_loss(
            spec_1, spec_2, reduction="mean") + F.l1_loss(
                torch.log(spec_1), torch.log(spec_2), reduction="mean")
    return loss
