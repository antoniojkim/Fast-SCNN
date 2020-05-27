# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import torch
import torchx
from PIL import Image
from torchvision import transforms

    
paths = set(p for p in os.listdir("data") if p.startswith("2018"))
val = {"20181204_135952", "20181107_132300"}
test = {"20181204_191844"}
datasets = {
    "train": paths - val - test,
    "val": val,
    "test": test
}

class AudiDataset:

    classes = (
        "Sidebars",
        "Curbstone",
        "Solid line",
        "Zebra crossing",
        "Dashed line",
        "Painted driv. instr.",
    )

    def __init__(
        self,
        crop_height,
        crop_width,
        resize_scale,
        mode="train",
        dataset_path="data",
        single_class=False,
    ):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.resize_scale = resize_scale
        self.mode = mode
        self.dataset_path = dataset_path

        with open(os.path.join(dataset_path, "class_list.json")) as file:
            class_list = json.loads(file.read())
            class_list = {v: k for k, v in class_list.items()}
            if single_class:
                self.class_list = [
                    (torchx.utils.hex_to_rgb(class_list[c]), 1)
                    for i, c in enumerate(self.classes)
                ]
            else:
                self.class_list = [
                    (torchx.utils.hex_to_rgb(class_list[c]), i + 1)
                    for i, c in enumerate(self.classes)
                ]
                
        self.images = [
            os.path.join(self.dataset_path, date, "camera", "cam_front_center", image).replace("camera", "{type}")
            for date in datasets[self.mode]
            for image in os.listdir(os.path.join(self.dataset_path, date, "camera", "cam_front_center"))
        ]

        self.normalization_mean = (0.485, 0.456, 0.406)
        self.normalization_std = (0.229, 0.224, 0.225)

        # normalization
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_mean, self.normalization_std),
            ]
        )

        self.scale = (0.5, 1, 1.25, 1.5, 1.75, 2)

    def __len__(self):
        return len(self.images)
    
    def get_path(self, index):
        return (
           self.images[index].replace("{type}", "camera"),
           self.images[index].replace("{type}", "label")
        )
    
    def resized_crop(self, image):
        return transforms.functional.resized_crop(
            image,
            image.size[1] - self.crop_height,
            (image.size[0] - self.crop_width) // 2,
            self.crop_height,
            self.crop_width,
            (
                int(self.crop_height * self.resize_scale),
                int(self.crop_width * self.resize_scale),
            ),
        )

    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        image_path, label_path = self.get_path(index)
        
        image = self.resized_crop(Image.open(image_path))
        label = self.resized_crop(Image.open(label_path))

        if self.mode == "train" and np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        return (
            self.normalize(image).float(),
            # One hot encode for cross entropy loss
             torch.from_numpy(
                 torchx.utils.encode_array(np.array(label), self.class_list).astype(np.uint8)
            ).long()
        )

    def normalize(self, image):
        return self.to_tensor(image)

    def denormalize(self, image):
        denormalized = image.new(*image.size())
        for i, (mean, std) in enumerate(
            zip(self.normalization_mean, self.normalization_std)
        ):
            denormalized[i, :, :] = image[i, :, :] * std + mean

        return denormalized
