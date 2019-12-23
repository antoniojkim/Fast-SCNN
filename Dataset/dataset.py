
import torch
from torchvision import transforms

import numpy as np
from PIL import Image

import random

import os
import csv

def one_hot_encode(label, class_dict):
    encoded = np.zeros(label.shape[:-1])
    for index, colour in enumerate(class_dict.values()):
        encoded[np.where(np.all(label == colour, axis=2))] = index
        
    return encoded


class Dataset:
    
    def __init__(self, dataset_path, crop_height, crop_width, mode="train"):
        self.dataset_path = dataset_path
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.mode = mode
        
        with open(os.path.join(dataset_path, "class_dict.csv")) as file:
            rows = [row.strip().split(",") for row in file]
            self.class_dict = {}
            for row in rows[1:]:
                if int(row[4]) == 1:  # class_11
                    self.class_dict[row[0]] = tuple(map(int, row[1:-1]))
                    
        self.images = list(set(os.listdir(os.path.join(dataset_path, mode))) &
                           set(os.listdir(os.path.join(dataset_path, f"{mode}_labels"))))
        
        # normalization
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.scale = (0.5, 1, 1.25, 1.5, 1.75, 2)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        assert 0 <= index < len(self.images)
        
        seed = np.random.randint(2147483647)
        
        scale = np.random.choice(self.scale)
        scale = (int(self.crop_height * scale),
                 int(self.crop_width * scale))
                
            
        image = Image.open(os.path.join(self.dataset_path, self.mode, self.images[index]))
        
        if self.mode == "train":
            image = transforms.Resize(scale, Image.BILINEAR)(image)
            random.seed(seed)
            image = transforms.RandomCrop((self.crop_height, self.crop_width), pad_if_needed=True)(image)
            
        else:
            image = transforms.functional.crop(image, image.size[0] - self.crop_height,
                                                      (image.size[1] - self.crop_width) // 2,
                                                      self.crop_height, self.crop_width)
        
        
        label = Image.open(os.path.join(self.dataset_path, f"{self.mode}_labels", self.images[index]))
        
        if self.mode == "train":
            label = transforms.Resize(scale, Image.BILINEAR)(label)
            random.seed(seed)
            label = transforms.RandomCrop((self.crop_height, self.crop_width), pad_if_needed=True)(label)
            
        else:
            label = transforms.functional.crop(label, label.size[0] - self.crop_height,
                                                      (label.size[1] - self.crop_width) // 2,
                                                      self.crop_height, self.crop_width)
            
                    
        if self.mode == "train"and np.random.random() > 0.5:
            image = transforms.RandomHorizontalFlip(p=1)(image)
            label = transforms.RandomHorizontalFlip(p=1)(label)
            
            
        image = self.to_tensor(image).float();
        label = np.array(label)
        
        # One hot encode for cross entropy loss
        label = one_hot_encode(label, self.class_dict).astype(np.uint8)
        # label = label.astype(np.float32)
        label = torch.from_numpy(label).long()
        
        return image, label
        