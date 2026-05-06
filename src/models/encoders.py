from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoTokenizer, T5EncoderModel


class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.byt5 = T5EncoderModel.from_pretrained("google/byt5-small")

        self.byt5.eval()
        self.byt5.requires_grad_(False)

        self.dimensions = 1472

    def transform(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    def forward(self, x):
        # May somehow swap dimensions before IDK
        with torch.no_grad():
            outputs = self.byt5(
                input_ids=x["input_ids"], attention_mask=x["attention_mask"]
            )
        return outputs.last_hidden_state


class StyleEncoder(nn.Module):
    def __init__(self, label_cnt: int, dims = 64):
        super().__init__()
        self.embedder = nn.Embedding(label_cnt + 1, dims)
        self.none = label_cnt
        self.dimensions = dims

    def forward(self, x):
        return self.embedder(x)


# class StyleEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
#         self.dimensions = 512

#         self.resnet.eval()
#         self.resnet.requires_grad_(False)

#     def forward(self, x):
#         self.resnet.eval()
#         with torch.no_grad():
#             return torch.flatten(self.resnet(x), 1)


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
