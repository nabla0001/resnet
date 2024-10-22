import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Union

def evaluate_error(model: nn.Module,
                   data_loader: DataLoader,
                   device: str,
                   n: Optional[int] = None) -> float:
    """Evaluates error on n batches (default: n=len(data_loader))"""
    model.eval()

    n = len(data_loader) if n is None else n

    with torch.no_grad():
        error = 0
        total = 0
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            error += (predicted != labels).sum().item()

            if (i + 1) == n:
                break

    return error/total

import pickle
import pathlib

def save_experiment(experiment: dict, filepath: Union[str, pathlib.PosixPath]) -> None:

    with open(filepath, 'wb') as f:
        pickle.dump(experiment, f)