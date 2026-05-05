from typing import Optional

import torch
from torch import nn
from torch._decomp.decompositions import dropout
from torchvision.models import ResNet18_Weights, resnet18
from transformers import RobertaModel, RobertaTokenizer


class ContentEncoder(nn.Module):
    def __init__(self, token_dims = 64, token_cnt = 8):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.embedder = nn.Embedding(len(self.tokenizer), token_dims)
        self.token_cnt = token_cnt

    def transform(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    def forward(self, x):
        ids = self.tokenizer(x, padding=True, truncation=True, return_attention_mask=False, max_length=self.token_cnt, return_tensors="pt").to(self.device)["input_ids"]
        # May somehow swap dimensions before IDK
        emb = self.embedder(ids)
        return emb.view(len(x), -1)


class StyleEncoder(nn.Module):
    def __init__(self, label_cnt: int, dims = 64):
        super().__init__()
        self.embedder = nn.Embedding(label_cnt + 1, dims)
        self.none = label_cnt

    def forward(self, x):
        return self.embedder(x)


class LabelEncoder(nn.Module):
    def __init__(self, dropout_prob: float, label_cnt, output_dim = 576):
        super().__init__()
        self.style_enc = StyleEncoder(label_cnt)
        self.content_enc = ContentEncoder()
        self.projection = nn.Linear(576, output_dim)
        self.dropout_prob = dropout_prob

    def initialize_weights(self):
        nn.init.normal_(self.projection.weight, std=0.02)

    def token_drop(self, style, content, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.Tensor()
        if force_drop_ids is None:
            rands = torch.rand(style.shape[0], device=style.device)
            drop_ids = rands < self.dropout_prob
        else:
            drop_ids = torch.Tensor(force_drop_ids == 1)

        for i in range(drop_ids.shape[0]):
            if drop_ids[i]:
                content[i] = ""

        none_style = torch.ones_like(style, device=style.device)
        style = torch.where(
            drop_ids.unsqueeze(1), torch.full(style.shape, self.style_enc.none), style
        )
        return style, content

    def forward(self, style, content, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            style, content = self.token_drop(style, content, force_drop_ids)

        style = self.style_enc(style)
        content = self.content_enc(content)

        labels = torch.cat([style, content], dim=1)
        return self.projection(labels)
