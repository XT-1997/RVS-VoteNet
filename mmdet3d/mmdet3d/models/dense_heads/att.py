from email import message
from turtle import forward

from matplotlib.pyplot import xkcd
import torch
from torch.nn import Module, Dropout
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d

class Self_att(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(Self_att, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class Ground_att(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(Ground_att, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs, L, _ = x.size()
        bs, S, _ = source.size()
        query, key, value = x, source, source

        # Euclidean distance
        query = self.q_proj(query).unsqueeze(dim=2).repeat(1,1,S,1)
        key = self.k_proj(key).unsqueeze(dim=1).repeat(1,L,1,1)
        value = self.v_proj(value)

        message = torch.softmax(torch.norm(query - key, dim=3), dim=2)
        message = torch.matmul(message, value)

        message = self.merge(message) # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class Render_att(nn.Module):
    def __init__(self,stride):
        super(Render_att, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(256,256, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Conv1d(256,256,kernel_size=1)

        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=stride,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = BatchNorm1d(256)

    def forward(self, x, s):

        query, key, value = x, s, s

        w = self.global_pool(key)
        w = self.conv1(w)
        w = torch.sigmoid(self.conv2(w))
        message = query * w

        message = self.relu(self.bn(self.conv3(value)) + message) 

        return message



if __name__ == '__main__':
    seed_point1 = torch.randn(4, 1024, 256)
    seed_point2 = torch.randn(4, 512, 256)
    seed_point3 = torch.randn(4, 256, 256)

    model1 = Self_att(256, 4)
    model2 = Self_att(256, 4)
    model3 = Self_att(256, 4)

    seed_point1_1 = model1(seed_point1, seed_point1)
    seed_point2_2 = model2(seed_point2, seed_point2)
    seed_point3_3 = model1(seed_point3, seed_point3)
    print(seed_point1_1.shape, seed_point2_2.shape, seed_point3_3.shape)

    model4 = Ground_att(256, 4)
    model5 = Ground_att(256, 4)
    model6 = Ground_att(256, 4)

    seed_point3_2 = model4(seed_point2, seed_point3)
    seed_point3_1 = model5(seed_point1, seed_point3)
    seed_point2_1 = model6(seed_point1, seed_point2)
    print(seed_point3_2.shape, seed_point3_1.shape, seed_point2_1.shape)

    model7 = Render_att(stride=2)
    model8 = Render_att(stride=2)
    model9 = Render_att(stride=4)

    seed_point2_3 = model7(seed_point3.transpose(1,2), seed_point2.transpose(1,2)).transpose(1,2)
    seed_point1_2 = model8(seed_point2.transpose(1,2), seed_point1.transpose(1,2)).transpose(1,2)
    seed_point1_3 = model9(seed_point3.transpose(1,2), seed_point1.transpose(1,2)).transpose(1,2)
    print(seed_point2_3.shape, seed_point1_2.shape, seed_point1_3.shape)
