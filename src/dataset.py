import os
import random
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset


def augmentation(mode='train'):
    if mode == 'train':
        transform = [
            albu.HorizontalFlip(),
            # albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensor(),
        ]

    elif mode == 'valid' or mode == 'test':
        transform = [
            # albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensor(),
        ]

    return albu.Compose(transform)


class PokemonTrainDataset(Dataset):
    def __init__(self, img_ids_A, img_ids_B, img_A_dir, img_B_dir, size=256, transform=None):
        self.img_ids_A = img_ids_A
        self.img_ids_B = img_ids_B
        self.img_A_dir = img_A_dir
        self.img_B_dir = img_B_dir
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return len(self.img_ids_A)
    
    def __getitem__(self, idx):
        id_A = self.img_ids_A[idx][:-4]
        path_A = os.path.join(self.img_A_dir, id_A + '.png')
        img_A = cv2.imread(path_A)
        img_A = cv2.resize(img_A, (self.size, self.size))
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        id_B = self.img_ids_B[random.randint(0, len(self.img_ids_B) - 1)][:-4]
        path_B = os.path.join(self.img_B_dir, id_B + '.png')
        img_B = cv2.imread(path_B)
        img_B = cv2.resize(img_B, (self.size, self.size))
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented_A = self.transform(image=img_A)
            img_A = augmented_A['image']

            augmented_B = self.transform(image=img_B)
            img_B = augmented_B['image']

        return (img_A, img_B), (id_A, id_B)
