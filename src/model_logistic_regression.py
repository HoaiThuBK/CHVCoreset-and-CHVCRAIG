"""
---file logistic_regression.py---
Logistic Regression model for multi-class classification:
- Input: feature matrix X (each sample is represented as a flattened
  feature vector).
- Output: logits = WX + b.

"""

import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Với Logistic Regression embedding trước classifier chính là vector đầu vào x"""
        return x
    
