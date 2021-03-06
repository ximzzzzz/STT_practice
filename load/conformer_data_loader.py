import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler

import librosa
import numpy as np
import scipy.signal
import torch
from scipy.io.wavfile import read
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
sys.path.append('./data')
from .spec_augment import spec_augment
import pandas as pd
import traceback

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data

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
    sample_rate, sound = read(path)
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
        if self.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # SFFT
#         D = librosa.stft(y, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = self.window)
        S = librosa.feature.melspectrogram(y, 
                                       sr = self.sample_rate, 
                                       n_fft = n_fft, 
                                       hop_length = hop_length, 
                                       win_length = win_length, 
                                       window= self.window,
                                       n_mels=80
                                      )
        log_S = librosa.power_to_db(S, ref=np.max)
#         spect, phase = librosa.magphase(D)
#         S = log(S+1)
#         spect = np.log1p(spect)
        spect = torch.FloatTensor(log_S)
#         print(f'spect shape {spect.shape}')
        
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
            
        if self.spec_augment:
            spect = spec_augment(spect)
        
        return spect
    
    
class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, speed_volume_perturb=False, spec_augment=False):
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = np.array(ids)
        self.size = len(ids)
        self.label_map = dict([(labels[i], i ) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)

        
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_str = sample[0], sample[1]
#         print(f'audio path : {audio_path}')
#         print(f'transcript : {transcript_str}')
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_str)

        return spect, transcript
    
    def parse_transcript(self, transcript_str):

        transcript = list(filter(None, [self.label_map.get(x) for x in list(transcript_str)]))

        return transcript
    
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
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        try:
            y = load_audio(augmented_filename)
        except :
            print(f'fuckig path : {path}')
            traceback.print_exc()
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
