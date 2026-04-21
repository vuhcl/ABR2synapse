import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from utils.ABRA_35.models import interpolate_and_smooth

time_scale = 20


def augment_waveform(waveform, augmentation_type="noise", **kwargs):
    """
    Apply various augmentations to waveform data
    """
    waveform = np.array(waveform)

    if augmentation_type == "noise":
        # Add Gaussian noise
        noise_level = kwargs.get("noise_level", 0.02)
        noise = np.random.normal(0, noise_level * np.std(waveform), waveform.shape)
        return waveform + noise

    elif augmentation_type == "time_shift":
        # Circular time shift
        shift_range = kwargs.get("shift_range", 5)
        shift = np.random.randint(-shift_range, shift_range + 1)
        return np.roll(waveform, shift)

    elif augmentation_type == "amplitude_scale":
        # Scale amplitude
        scale_range = kwargs.get("scale_range", (0.9, 1.1))
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return waveform * scale

    elif augmentation_type == "baseline_drift":
        # Add slow baseline drift
        drift_amplitude = kwargs.get("drift_amplitude", 0.01)
        t = np.linspace(0, 2 * np.pi, len(waveform))
        drift = drift_amplitude * np.sin(t * np.random.uniform(0.5, 2.0))
        return waveform + drift

    return waveform


time_scale = 20


class WaveformDataset(Dataset):
    def __init__(
        self, df, data="waveform", augment=False, augment_prob=0.7, feature_pred=False
    ):
        self.df = df
        self.data = data
        self.augment = augment
        self.augment_prob = augment_prob
        self.feature_pred = feature_pred
        if self.feature_pred:
            self.waveforms, self.targets, self.amplitudes, self.noises = (
                self.prepare_data(df)
            )
        else:
            self.waveforms, self.targets = self.prepare_data(df)

    def prepare_data(self, df):
        waveforms = []
        targets = []
        amplitudes = []
        noises = []

        for idx, row in df.iterrows():
            if self.data == "waveform":
                orig_y = row["Waveform"]
                waveform = self.preprocess_waveform(orig_y, time_scale)
            elif self.data == "wavei":
                waveform = row["WaveI"]

            waveforms.append(waveform)
            targets.append(row["SynapsesPerIHC"])
            amplitudes.append(row["Amplitude"])
            noises.append(row["Noise"])

        if self.feature_pred:
            return waveforms, targets, amplitudes, noises
        else:
            return waveforms, targets

    def preprocess_waveform(self, waveform, time_scale):
        orig_y = waveform
        tenms = int((10 / time_scale) * len(orig_y)) if time_scale > 10 else len(orig_y)
        return interpolate_and_smooth(orig_y[:tenms], 244)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx].copy()

        # Apply random augmentation during training
        if self.augment and np.random.random() < self.augment_prob:
            # Apply multiple augmentations with some probability
            if np.random.random() < 0.3:  # 30% chance of noise
                waveform = augment_waveform(waveform, "noise", noise_level=0.015)
            if np.random.random() < 0.3:  # 30% chance of time shift
                waveform = augment_waveform(waveform, "time_shift", shift_range=3)
            if np.random.random() < 0.2:  # 20% chance of amplitude scaling
                waveform = augment_waveform(
                    waveform, "amplitude_scale", scale_range=(0.95, 1.05)
                )
            if np.random.random() < 0.1:  # 10% chance of baseline drift
                waveform = augment_waveform(
                    waveform, "baseline_drift", drift_amplitude=0.005
                )

        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.feature_pred:
            amplitude_tensor = torch.tensor(self.amplitudes[idx], dtype=torch.float32)
            noise_tensor = torch.tensor(self.noises[idx], dtype=torch.float32)
            return waveform_tensor, target_tensor, amplitude_tensor, noise_tensor
        return waveform_tensor, target_tensor
