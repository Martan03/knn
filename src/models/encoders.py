from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import RobertaModel, RobertaTokenizer


class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dimensions = 768

    def transform(self, text):
        return self.tokenizer(text, padding=True, truncation=True, returns_tensors="pt")

    def forward(self, x):
        # May somehow swap dimensions before IDK
        outputs = self.roberta(
            inputs_ids=x["input_ids"], attention_mask=x["attention_mask"]
        )
        return outputs.avg(axis=1)


class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimensions = 512

    def forward(self, x):
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

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.Tensor()
        if force_drop_ids is None:
            rands = torch.rand(labels.shape[0], device=labels.device)
            drop_ids = rands < self.dropout_prob
        else:
            drop_ids = torch.Tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.none_label, labels)
        return labels

    def forward(self, style, content, train, force_drop_ids=None):
        style = self.style_enc(style)
        content = self.content_enc(content)

        use_dropout = self.dropout_prob > 0
        labels = style.cat(content)
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.projection(labels)
