import math
import torch
from torch import nn


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


class CrossModalAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        # 计算注意力分数
        g = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attention_size)
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output


# 类门控机制与线性融合，原方法
class FusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(FusionLayer, self).__init__()
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_new, h_feature):
        gate = self.sigmoid(self.linear(torch.cat([h_new, h_feature], dim=-1)))
        h_combined = gate * h_new + (1 - gate) * h_feature
        return h_combined


class GatedFusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusionLayer, self).__init__()
        self.linear_h_d = nn.Linear(hidden_size, hidden_size)
        self.linear_h_z = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, h_d, h_z):
        gate = self.sigmoid(self.linear_h_d(h_d) + self.linear_h_z(h_z))
        fused_output = gate * h_d + (1 - gate) * h_z
        return self.activation(fused_output)


class BilinearFusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(BilinearFusionLayer, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, h_d, h_z):
        fused_output = self.bilinear(h_d, h_z)
        return self.activation(fused_output)


class WeightedSumFusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(WeightedSumFusionLayer, self).__init__()
        self.weight_h_d = nn.Parameter(torch.randn(hidden_size))
        self.weight_h_z = nn.Parameter(torch.randn(hidden_size))
        self.activation = nn.Tanh()

    def forward(self, h_d, h_z):
        # 加权求和
        weighted_h_d = h_d * self.weight_h_d
        weighted_h_z = h_z * self.weight_h_z
        fused_output = weighted_h_d + weighted_h_z
        return self.activation(fused_output)


# 多余批量大小维度，进行处理后才可继续使用。对h_d,h_z进行融合
# class MultiHeadAttentionFusionLayer(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super(MultiHeadAttentionFusionLayer, self).__init__()
#         self.multi_head_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
#         self.activation = nn.Tanh()
#
#     def forward(self, h_d, h_z):
#         # 将h_d和h_z合并成一个张量作为query和key
#         query = torch.cat([h_d.unsqueeze(0), h_z.unsqueeze(0)], dim=0)
#         key = torch.cat([h_d.unsqueeze(0), h_z.unsqueeze(0)], dim=0)
#         value = torch.cat([h_d.unsqueeze(0), h_z.unsqueeze(0)], dim=0)
#
#         fused_output, _ = self.multi_head_attention(query=query, key=key, value=value)
#         fused_output = fused_output.squeeze(0)
#         return self.activation(fused_output)


class AttentionFusionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionFusionLayer, self).__init__()
        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, h_new, h_feature, value):
        Q = self.query_linear(h_feature)
        K = self.key_linear(h_new)
        V = self.value_linear(value)

        g = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        scores = torch.softmax(g, dim=-1)
        output = torch.matmul(scores, V)

        return output


class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output
