import torch.nn as nn
import torch.nn.functional as F


def entropy_loss(x, ignore_mask):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = b.sum(dim=2)
    b[ignore_mask] = 0 # Mask after sum to avoid memory issue.
    b = -1.0 * b.sum(dim=0).mean() # Sum along words and mean along batch
    return b

