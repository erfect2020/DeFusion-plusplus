import torch
from torch import nn
import torch.nn.functional as F

class MultiInputSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

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

        assert not torch.any(torch.isnan(q)), "MultiInputSelfAttentionVpaper query nan"
        if self.common:
            k = self.key(x2)
        else:
            k = self.key(x1)
        assert not torch.any(torch.isnan(k)), "MultiInputSelfAttentionVpaper key nan"
        v = self.value(x1)
        assert not torch.any(torch.isnan(v)), "MultiInputSelfAttentionVpaper value nan"

        # mask = (k.sum(dim=0) == 0).unsqueeze(0).expand(q.size(0), -1)
        # mask = mask * -1e9  # ? True ????????????
        def check_and_fill_nan(tensor):
            if torch.isnan(tensor).any():
                print("Warning: NaN values found. Replacing with zeros.")
                tensor = torch.nan_to_num(tensor)
            if torch.isinf(tensor).any():
                print("Warning: Inf values found. Replacing with zeros.")
                tensor = torch.inf_to_num(tensor)
            return tensor

        q = check_and_fill_nan(q)
        k = check_and_fill_nan(k)
        v = check_and_fill_nan(v)

        # print("k shape", k.shape)
        # mask = (abs(k.sum(dim=-1)) < 1e-8)  #
        # mask = mask.transpose(0, 1)

        # attn = torch.bmm(q, k.transpose(-2, -1))
        # assert not torch.any(torch.isnan(attn)), "MultiInputSelfAttentionVpaper attn nan"
        # attn = F.softmax(attn, dim=-1)
        # assert not torch.any(torch.isnan(attn)), "MultiInputSelfAttentionVpaper softmax nan"

        # for i in range(197):
        #     print(mask[i], torch.any(mask[i] == True))
        # mask = mask.unsqueeze(0).unsqueeze(0).expand(q.size(0), -1, -1)
        # mask = mask * -1e9  # ? True ????????????

        attn_output, _ = self.multihead_attn(q, k, v)
        assert not torch.any(torch.isnan(attn_output)), "MultiInputSelfAttentionVpaper multihead nan"
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
        assert not torch.any(torch.isnan(src)), "TransformerAttenBlockVpaper multi attention nan"
        src = self.norm1(src)

        assert not torch.any(torch.isnan(src)), "TransformerAttenBlockVpaper norm1 nan"

        src = src + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src)))))
        src = self.norm2(src)

        assert not torch.any(torch.isnan(src)), "TransformerAttenBlockVpaper norm2 nan"
        return src
