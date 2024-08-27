import torch
import torch.nn as nn
import torch.nn.functional as F

from baseModel import PositionalEncoding, GCN, moduleATT_softmax, cosine_func

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class xLSTM(nn.Module):
    def __init__(self, embedding_dim, context_length, num_blocks=7, num_heads=4):
        super().__init__()
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=[1]
        )
        self.model = xLSTMBlockStack(cfg)

    def forward(self, x):
        return self.model(x)


class TripleModel_Catt(nn.Module):
    def __init__(self, dim=128, head=8, metric='cosine', p=0.2):
        super().__init__()
        video_neighbor = 3
        text_neighbor = 2 # UDIVA 设为10
        audio_neighbor = 3
        max_len = 13 # UDIVA 设为452
        act = nn.ReLU()

        self.video_xlstm = xLSTM(768, 15)
        self.video_lin_xlstm = nn.Sequential(nn.Dropout(p), nn.Linear(768, dim), act)

        self.video_position = PositionalEncoding(768, 15)
        self.video_gcn = nn.Sequential(GCN(768, video_neighbor, metric=metric), GCN(768, video_neighbor, metric=metric))
        self.video_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.text_position = PositionalEncoding(768, max_len)  # 数据集中文本句子的最大长度为13
        self.text_gcn = nn.Sequential(GCN(768, text_neighbor, metric=metric), GCN(768, text_neighbor, metric=metric))
        self.text_lin = nn.Sequential(nn.Dropout(p), nn.Linear(768 * 1, dim), act)

        self.wav2clip_position = PositionalEncoding(512, 15)
        self.wav2clip_gcn = nn.Sequential(GCN(512, audio_neighbor, metric=metric), GCN(512, audio_neighbor, metric=metric))
        self.wav2clip_lin = nn.Sequential(nn.Dropout(p), nn.Linear(512, dim), act)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 融合模型
        num = 3
        self.mutliHead = moduleATT_softmax(num=num, head=head)
        self.lin_m = nn.Sequential(nn.Linear(dim * head, dim), nn.ReLU(), nn.Dropout(p))

        self.mlp_m = nn.Sequential(nn.Linear(dim, 5), nn.Sigmoid())


    def forward(self, video, text, wav2clip):

        xlstm_v = self.video_xlstm(video)
        xlstm_v = self.video_lin_xlstm(xlstm_v).mean(dim=1)

        vpos = self.video_position(video)
        clip_v = video + vpos
        clip_v = self.video_gcn(clip_v)
        clip_v = self.video_lin(clip_v)
        clip_v = self.pooling(clip_v.permute(0, 2, 1)).squeeze(2)

        tpos = self.text_position(text)
        t = text + tpos
        t = self.text_gcn(t)
        t = self.text_lin(t)
        t = self.pooling(t.permute(0, 2, 1)).squeeze(2)

        a2pos = self.wav2clip_position(wav2clip)
        wav = wav2clip + a2pos
        wav = self.wav2clip_gcn(wav)
        wav = self.wav2clip_lin(wav)
        wav = self.pooling(wav.permute(0, 2, 1)).squeeze(2)

        clip_clip = cosine_func(clip_v, t)
        clip_wav = cosine_func(clip_v, wav)

        x = self.mutliHead(xlstm_v, clip_clip, clip_wav)
        x = self.lin_m(x)
        out = self.mlp_m(x)

        return out



if __name__ == '__main__':
    pass


