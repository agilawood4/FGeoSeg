import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerB0Sky(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b0-finetuned-ade-512-512',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    def forward(self, x):
        return self.model(pixel_values=x, labels=None)
