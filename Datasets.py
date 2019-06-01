from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob, os

class MyDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.getData(self.root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
  
    def getData(self, root):
        num_id = {'fake': 0, 'gan': 0, 'real': 1}
        dirs = glob.glob(os.path.join(root, '*/*.jpg'))
        images = [Image.open(dir) for dir in dirs]
        labels = [num_id[os.path.basename(os.path.dirname(dir))] for dir in dirs]

        return images, labels

class MyTestDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self.getData(self.root)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
  
    def getData(self, root):
        num_id = {'fake': 0, 'gan': 1, 'real': 2}
        dirs = glob.glob(os.path.join(root, '*/*.jpg'))
        images = [Image.open(dir) for dir in dirs]
        labels = [dir for dir in dirs]

        return images, labels
