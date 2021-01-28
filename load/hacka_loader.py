import os
import subprocess
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import wave
import math

import torch
import torch.nn as nn
from torch.distributed import get_rank, get_world_size
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

import librosa
import scipy.signal
from scipy.io.wavfile import read

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError

def load_audio(path):
#     sample_rate, sound = read(path)
    sample_rate = 16000
    with open(path, 'rb') as f:
        raw = f.read() 
        sound = np.frombuffer(raw, dtype='int16')
    sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    sound = 2*((sound - sound.min()) / (sound.max() - sound.min())) - 1
    return sound        

class SpectrogramParser(AudioParser):
    
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.speed_volume_perturb = speed_volume_perturb
        self.spec_augment = spec_augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')
        
    def parse_audio(self, audio_path):

        y = load_audio(audio_path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # SFFT
        D = librosa.stft(y, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        
        return spect
    
    
class SpecDatasetTest(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, path_list, labels, normalize=True, speed_volume_perturb=False, spec_augment=False):
        self.path_list = path_list
#         self.ids = ids
        self.size = len(path_list)
        self.label_map = dict([(labels[i], i ) for i in range(len(labels))])
        super(SpecDatasetTest, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)
        
    def __getitem__(self, index):
        audio_path = self.path_list[index]
        spect = self.parse_audio(audio_path)
        return spect, audio_path
    
    def __len__(self):
        return self.size
    
def _collate_fn(batch):
    def func(p):
        return p[0].size(1)
    batch = sorted(batch, key=lambda sample : sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size   =   longest_sample.size(0)
    max_seqlength = longest_sample.size(1)
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    path_list = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        path = sample[1]
#         print(path)
        path_list.append(path)
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
    return inputs, input_percentages, path_list

class AudioDataLoaderTest(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoaderTest, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn