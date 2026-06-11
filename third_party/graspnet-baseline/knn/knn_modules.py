import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
try:
  from knn_pytorch import knn_pytorch
except ImportError:
  knn_pytorch = None

def knn(ref, query, k=1):
  """ Compute k nearest neighbors for each query point.
  """
  device = ref.device
  ref = ref.float().to(device)
  query = query.float().to(device)
  if knn_pytorch is None:
    # ref/query are (B, C, N). Match the extension's 1-based index output.
    dist = torch.cdist(query.transpose(1, 2), ref.transpose(1, 2))
    return torch.topk(dist, k=k, dim=2, largest=False).indices.transpose(1, 2) + 1
  inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
  knn_pytorch.knn(ref, query, inds)
  return inds
