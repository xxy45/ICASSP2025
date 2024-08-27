import torch
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(float(self._data[-1]))

    @property
    def label(self):
        return torch.tensor([float(self._data[i]) for i in range(1, 6)])  # ocean

class myDataset(Dataset):
    def __init__(self, filepath, mod='train'):
        super().__init__()
        self.mod = mod
        self.csv_filepath = filepath
        self.max_text_len_seq = 13

        clip_image_path = f'data/{mod}_clipimage.pkl'
        with open(clip_image_path, 'rb') as f:
            self.video = pickle.load(f)   # 对抽取的15个视频帧使用clip提取的特征

        clip_text_path = f'data/{mod}_clipsentence.pkl'
        with open(clip_text_path, 'rb') as f:  # 使用clip-vit提取的句子信息，每个句子作为一个特征向量
            self.text = pickle.load(f)

        wav2clip_path = f"data/{mod}_15_audio_wav2clip.pkl"
        with open(wav2clip_path, 'rb') as f:
            self.wav2clip = pickle.load(f)

        self._parse_list()

    def _parse_list(self):
        tmp = [x.strip().split(',') for x in open(self.csv_filepath)]
        self.video_list = [VideoRecord(item) for item in tmp]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        record = self.video_list[index]

        v = self.video[record.path + '.mp4']
        v = torch.stack(v, dim=0).squeeze(0).to(torch.float32)  # [15, 768]
        v = torch.FloatTensor(v)

        wav2clip = self.wav2clip[record.path + '.mp4']
        wav2clip = torch.FloatTensor(wav2clip)  # [15,512]

        t = self.text[record.path + '.mp4']
        t = torch.stack(t, dim=0).to(torch.float32)   # [n, 768]  # 最大长度13
        t = torch.FloatTensor(t)
        t_len = t.shape[0]
        t = F.pad(t, (0, 0, 0, self.max_text_len_seq - t_len), "constant", 0)

        label = record.label
        return label, v, t, wav2clip


if __name__ == '__main__':
    pass
