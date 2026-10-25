"""Paragraph classifier: transformer encoder, mean pooling, one logit (>0 = AI).

Same layout as desklib/ai-text-detector-v1.01, so desklib weights load directly and can be fine-tuned.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

DESKLIB = "desklib/ai-text-detector-v1.01"


class MeanPoolClassifier(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


def new_classifier(base):
    """'desklib' continues from the desklib detector; any other hub id starts from that encoder."""
    if base == "desklib":
        return MeanPoolClassifier.from_pretrained(DESKLIB), AutoTokenizer.from_pretrained(DESKLIB)
    cfg = AutoConfig.from_pretrained(base)
    m = MeanPoolClassifier(cfg)
    m.model = AutoModel.from_pretrained(base)
    return m, AutoTokenizer.from_pretrained(base)


def load_classifier(path):
    return MeanPoolClassifier.from_pretrained(path), AutoTokenizer.from_pretrained(path)
