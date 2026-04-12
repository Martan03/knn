import torch
from torch import nn
from torchtext.functional import to_tensor
from torchtext.models import ROBERTA_BASE_ENCODER
from torchvision.models import ResNet18_Weights, resnet18


class ContentEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.roberta = ROBERTA_BASE_ENCODER.get_model()
        self.projection = nn.Linear(768, output_dim)

    def transform(self, text):
        assert ROBERTA_BASE_ENCODER.transform
        return to_tensor(ROBERTA_BASE_ENCODER.transform(text), padding_value=1)

    def forward(self, x):
        # May somehow swap dimensions before IDK
        return self.projection(self.roberta(x))


class StyleEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-1]))

        self.projection = nn.Linear(512, output_dim)

    def forward(self, x):
        features = torch.flatten(self.resnet(x), 1)
        return self.projection(features)


class LabelEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.style_enc = StyleEncoder(output_dim)
        self.content_enc = ContentEncoder(output_dim)

    def text_transform(self, text):
        self.content_enc.transform(text)

    def forward(self, style, content):
        # [Batch, 1, 768]
        style = self.style_enc(style).unsqueeze(1)
        # [Batch, Seq, 768]
        content = self.content_enc(content)
        # Maybe somehow swap dimensions before and use different dimension to
        # concat IDK
        # [Batch, Seq + 1, 768]
        return torch.cat([style, content], dim=1)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.labels = LabelEncoder(hidden_size)
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
