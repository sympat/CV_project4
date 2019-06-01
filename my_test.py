import argparse, pprint, os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import alexnet, resnet18, resnet50
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from Datasets import MyDataset


a = torch.ones(5, dtype=torch.int32)
masked = lambda t: 1 if t == 2 else 0
b = np.array([masked(element) for element in a])

print(a)
print(b)
