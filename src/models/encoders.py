from typing import Optional

import torch
import math
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoTokenizer, T5EncoderModel
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ContentEncoder(nn.Module):
    def __init__(self, dim=1472):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.embedder = nn.Embedding(len(self.tokenizer), dim)
        self.position = PositionalEncoding(dim)

        self.dimensions = dim
        
        
    def transform(self, text):
        return self.tokenizer(text, padding=True, return_tensors="pt")

    def forward(self, x):
        xe = self.embedder(x["input_ids"])
        xp = self.position(xe)
        return xp, x["attention_mask"]


class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimensions = 512

        self.resnet.eval()
        self.resnet.requires_grad_(False)

    def forward(self, x):
        self.resnet.eval()
        with torch.no_grad():
            return torch.flatten(self.resnet(x), 1)
            
            


class LabelEncoder(nn.Module):
    def __init__(self, dropout_prob: float, output_dim: Optional[int] = None):
        super().__init__()
        self.style_enc = StyleEncoder()
        self.content_enc = ContentEncoder()
        dims = self.style_enc.dimensions + self.content_enc.dimensions
        self.dimensions = output_dim if output_dim else dims
        self.none_label = torch.zeros(self.dimensions)
        self.projection = nn.Linear(dims, self.dimensions)
        self.dropout_prob = dropout_prob

    def text_transform(self, text, device):
        encoded = self.content_enc.transform(text)
        return {k: v.to(device) for k, v in encoded.items()}

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

        content = {
            k: torch.where(
                drop_ids.unsqueeze(1), torch.zeros_like(v, device=v.device), v
            )
            for k, v in content.items()
        }
        none_style = torch.ones_like(style, device=style.device)
        style = torch.where(
            drop_ids.unsqueeze(1).unsqueeze(1).unsqueeze(1), none_style, style
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
