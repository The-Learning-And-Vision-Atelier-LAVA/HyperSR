import torch
import torch.nn as nn
import torch.nn.functional as F


class ResB(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
        )
    def forward(self, x):
        return self.body(x) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(self.norm(x)) + x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super(MultiHeadAttention, self).__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, input_q, input_k, input_v):
        q = self.to_q(self.norm(input_q))       # b, h_q*w_q, c
        k = self.to_k(self.norm(input_k))       # b, h_k*w_k, c
        v = self.to_v(input_v)                  # b, h_v*w_v, c

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale        # b, h_q*w_q, h_k*w_k
        att = dots.softmax(-1)

        out = torch.matmul(att, v)

        return self.to_out(out) + input_q


class TransformerB(nn.Module):
    def __init__(self, dim, heads=1, dim_head=64, mlp_dim=64):
        super(TransformerB, self).__init__()
        self.MSA = MultiHeadAttention(dim, heads, dim_head)
        self.mlp = FeedForward(dim, mlp_dim)

    def forward(self, input):
        q, k, v = input
        out = self.MSA(q, k, v)
        out = self.mlp(out)

        return [out, k, v]


class HyperSR(nn.Module):
    def __init__(self, channels_LSI, channels_HSI, channels, n_endmembers=64):
        super(HyperSR, self).__init__()
        self.ini_spatial = nn.Conv2d(channels_LSI, channels, 3, 1, 1)
        self.ini_spectral = nn.Conv2d(channels_HSI, channels, 1, 1, 0)
        self.fea_spatial = nn.Sequential(
            ResB(channels),
            ResB(channels),
            ResB(channels),
            ResB(channels),
        )
        self.fea_spectral = nn.Sequential(
            ResB(channels, 1),
            ResB(channels, 1),
            ResB(channels, 1),
            ResB(channels, 1),
        )

        self.fea_endmember = nn.Parameter(torch.randn(1, n_endmembers, channels))
        self.encoder = nn.Sequential(
            TransformerB(channels),
            TransformerB(channels),
            TransformerB(channels)
        )
        self.decoder = nn.Sequential(
            TransformerB(channels),
            TransformerB(channels),
            TransformerB(channels)
        )
        self.tail = nn.Sequential(
            ResB(channels),
            ResB(channels),
            nn.Conv2d(channels, channels_HSI, 3, 1, 1)
        )

    def forward(self, HrLSI, LrHSI):
        # initial
        fea0_HrLSI = self.ini_spatial(HrLSI)
        fea0_LrHSI = self.ini_spectral(LrHSI)

        fea_HrLSI = self.fea_spatial(fea0_HrLSI) + fea0_HrLSI
        fea_LrHSI = self.fea_spectral(fea0_LrHSI) + fea0_LrHSI

        # body
        b, c, h_lr, w_lr = fea_LrHSI.shape
        b, c, h_hr, w_hr = fea_HrLSI.shape

        fea_LrHSI = fea_LrHSI.view(b, c, -1).transpose(-1, -2)
        fea_HrLSI = fea_HrLSI.view(b, c, -1).transpose(-1, -2)

        [fea_endmember, _, _] = self.encoder([self.fea_endmember, fea_LrHSI, fea_LrHSI])
        [out, _, _] = self.decoder([fea_HrLSI, fea_endmember, fea_endmember])

        # tail
        out = out.transpose(-1, -2).contiguous().view(b, -1, h_hr, w_hr)
        out = self.tail(out)

        return out
