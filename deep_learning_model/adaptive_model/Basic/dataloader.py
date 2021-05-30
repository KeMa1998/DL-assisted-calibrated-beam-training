import os
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import torch
import math
from collections import Counter

# detailed code comments can be seen in the folder basic_model
class Dataloader():
    def __init__(self, path='', batch_size=32, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for i, f in enumerate(self.files):
            if not f.split('.')[-1] == 'mat':
                del (self.files[i])
        self.reset()

    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]
        self.buffer1 = np.zeros((0, 2, 10, 16))
        self.buffer2 = np.zeros((0, 2, 10, 16))
        self.buffer_label_m = np.zeros((0, 10))
        self.buffer_label_nonoise_m = np.zeros((0, 10))
        self.buffer_beam_power_m = np.zeros((0, 10, 64))
        self.buffer_beam_power_nonoise_m = np.zeros((0, 10, 64))

    def load(self, file):
        data = sio.loadmat(file)
        channel1 = data['channel_data1']
        channel2 = data['channel_data2']
        channel1 = np.transpose(channel1, (1, 0, 2, 3))
        channel1 = channel1[:, :, :, 0 : 61 : 4]
        channel2 = np.transpose(channel2, (1, 0, 2, 3))
        labels = data['max_id_sery_m'] - 1
        labels_nonoise = data['max_id_sery_no_noise_m'] - 1
        beam_power_m = data['rsrp_sery_m']
        beam_power_nonoise_m = data['rsrp_sery_no_noise_m']
        return channel1, channel2, labels, beam_power_m, beam_power_nonoise_m, labels_nonoise

    def pre_process(self, channels):
        return channels

    def next_batch(self):
        done = False
        count = True
        while self.buffer1.shape[0] < self.batch_size:
            if len(self.unvisited_files) == 0:
                done = True
                count = False
                break
            channel1, channel2, labels, beam_power_m, beam_power_nonoise_m, labels_nonoise = self.load(
                self.unvisited_files.pop(0))

            del self.buffer1
            del self.buffer2
            del self.buffer_label_m
            del self.buffer_beam_power_m
            del self.buffer_beam_power_nonoise_m
            del self.buffer_label_nonoise_m

            self.buffer1 = np.zeros((0, 2, 10, 16))
            self.buffer2 = np.zeros((0, 2, 10, 16))
            self.buffer_label_m = np.zeros((0, 10))
            self.buffer_beam_power_m = np.zeros((0, 10, 64))
            self.buffer_beam_power_nonoise_m = np.zeros((0, 10, 64))
            self.buffer_label_nonoise_m = np.zeros((0, 10))

            self.buffer1 = np.concatenate((self.buffer1, channel1), axis=0)
            self.buffer2 = np.concatenate((self.buffer2, channel2), axis=0)
            self.buffer_label_m = np.concatenate((self.buffer_label_m, labels), axis=0)
            self.buffer_beam_power_m = np.concatenate((self.buffer_beam_power_m, beam_power_m), axis=0)
            self.buffer_beam_power_nonoise_m = np.concatenate((self.buffer_beam_power_nonoise_m, beam_power_nonoise_m), axis=0)
            self.buffer_label_nonoise_m = np.concatenate((self.buffer_label_nonoise_m, labels_nonoise), axis=0)

        out_size = min(self.batch_size, self.buffer1.shape[0])
        batch_channel1 = self.buffer1[0:out_size, :, :, :]
        batch_channel2 = self.buffer2[0:out_size, :, :, :]
        batch_labels_m = np.squeeze(self.buffer_label_m[0:out_size, :])
        batch_beam_power_m = np.squeeze(self.buffer_beam_power_m[0:out_size, :, :])
        batch_beam_power_nonoise_m = np.squeeze(self.buffer_beam_power_nonoise_m[0:out_size, :, :])
        batch_labels_nonoise_m = np.squeeze(self.buffer_label_nonoise_m[0:out_size, :])

        self.buffer1 = np.delete(self.buffer1, np.s_[0:out_size], 0)
        self.buffer2 = np.delete(self.buffer2, np.s_[0:out_size], 0)
        self.buffer_label_m = np.delete(self.buffer_label_m, np.s_[0:out_size], 0)
        self.buffer_beam_power_m = np.delete(self.buffer_beam_power_m, np.s_[0:out_size], 0)
        self.buffer_beam_power_nonoise_m = np.delete(self.buffer_beam_power_nonoise_m, np.s_[0:out_size], 0)
        self.buffer_label_nonoise_m = np.delete(self.buffer_label_nonoise_m, np.s_[0:out_size], 0)

        batch_channel1 = np.float32(batch_channel1)
        batch_channel2 = np.float32(batch_channel2)
        batch_labels_m = batch_labels_m.astype(long)
        batch_beam_power_m = np.float32(batch_beam_power_m)
        batch_beam_power_nonoise_m = np.float32(batch_beam_power_nonoise_m)
        batch_labels_nonoise_m = batch_labels_nonoise_m.astype(long)

        return torch.from_numpy(batch_channel1).to(self.device), torch.from_numpy(batch_channel2).to(
            self.device), torch.from_numpy(batch_labels_m).to(
            self.device), torch.from_numpy(batch_beam_power_m).to(
            self.device), torch.from_numpy(batch_beam_power_nonoise_m).to(
            self.device), torch.from_numpy(batch_labels_nonoise_m).to(
            self.device), done, count