import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiInputSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attn_output, _ = self.multihead_attn(q, k, v)
        return attn_output


class MultiInputSelfAttentionVpaper(nn.Module):
    def __init__(self, d_model, nhead, common):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.common = common

    def forward(self, x1, x2):
        q = self.query(x2)
        if self.common:
            k = self.key(x2)
        else:
            k = self.key(x1)
        v = self.value(x1)
        attn_output, _ = self.multihead_attn(q, k, v, average_attn_weights=False)
        return attn_output


class TransformerAttenBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multi_input_self_attn = MultiInputSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src1, src2):
        src1 = src1 + self.dropout1(self.multi_input_self_attn(src1, src2))
        src1 = self.norm1(src1)

        src1 = src1 + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src1)))))
        src1 = self.norm2(src1)

        return src1


class TransformerAttenBlockVpaper(nn.Module):
    def __init__(self, d_model, nhead, common, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multi_input_self_attn = MultiInputSelfAttentionVpaper(d_model, nhead, common=common)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src1, src2):
        src = src2 + self.dropout1(self.multi_input_self_attn(src1, src2)) # query is x2
        src = self.norm1(src)

        src = src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src)))))
        src = self.norm2(src)
        return src
