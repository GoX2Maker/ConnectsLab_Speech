from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as transforms
import torch
import torch.nn as nn

class AudioTransform():
    def __init__(self,device, sample_rate, n_mfcc, window_size, hop_length, desired_frames):
        self.device = device
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.desired_frames = desired_frames
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.desired_frames = desired_frames
        self.n_fft = int(sample_rate * window_size)
        self.mfcc_transform = transforms.MFCC(
                                                    sample_rate=self.sample_rate,
                                                    n_mfcc=self.n_mfcc,
                                                    melkwargs={
                                                        "n_fft": self.n_fft,
                                                        "hop_length": self.hop_length,
                                                        "mel_scale": "htk",
                                                    },
                                              )
        
    def getMFCC(self, audioPaths):
        waveform, sample_rate = torchaudio.load(audioPaths, normalize=True)

        # Desired number of frames in Mel spectrogram
        num_samples = self.desired_frames * self.hop_length

        # Trim or pad the waveform
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]  # Trim
        elif waveform.shape[1] < num_samples:
            pad_size = num_samples - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad_size))  # Pad

        mfcc_specgram = self.mfcc_transform(waveform)

        # If still not the correct shape, trim off extra samples
        if mfcc_specgram.shape[2] > self.desired_frames:
            mfcc_specgram = mfcc_specgram[:, :, :self.desired_frames]


        # Assuming mfcc is already computed
        mfcc_mean = mfcc_specgram.mean(dim=2, keepdim=True)
        mfcc_std = mfcc_specgram.std(dim=2, keepdim=True)

        # Avoid division by zero
        mfcc_std = mfcc_std.clamp(min=1e-5)

        # Normalize the MFCC
        normalized_mfcc = (mfcc_specgram - mfcc_mean) / mfcc_std

        return normalized_mfcc.to(self.device)


class AudioDataset(Dataset):
    def __init__(self, audioPaths, labels, device, sample_rate, n_mfcc, window_size, hop_length, desired_frames):
        self.device = device
        self.audioPaths = audioPaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.desired_frames = desired_frames
        self.n_mfcc = n_mfcc
        self.n_fft = int(sample_rate * window_size)
        self.audio_transform = AudioTransform(device, sample_rate, n_mfcc, window_size, hop_length, desired_frames)


    def __getitem__(self, index):
        mfcc =  self.audio_transform.getMFCC(self.audioPaths[index])
        
        return mfcc.to(self.device), self.labels[index]

    def __len__(self):
        return len(self.audioPaths)
    
