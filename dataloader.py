import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class Data(Dataset):
    def __init__(self, transform):
        self.transform = transform
        # self.data_dir --> replace your train data path 
        self.data_dir = './real'
        self.file_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.file_list[idx])
        img = Image.open(img_path)
        ground_truth = self.transform(img)

        return ground_truth

def data_loader(batch_size=16, shuffle=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))
                                   ])

    dset = Data(transform)

    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          drop_last=True)

    return dloader
