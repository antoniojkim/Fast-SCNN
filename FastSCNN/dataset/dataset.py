# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import torch
import torchx
from PIL import Image
from torchvision import transforms


class Dataset:

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

        self.images = list(
            set(os.listdir(os.path.join(dataset_path, mode)))
            & set(os.listdir(os.path.join(dataset_path, f"{mode}_labels")))
        )

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

    def __getitem__(self, index):
        assert 0 <= index < len(self.images)

        image = Image.open(
            os.path.join(self.dataset_path, self.mode, self.images[index])
        )
        image = transforms.functional.resized_crop(
            image,
            image.size[1] - self.crop_height,
            (image.size[0] - self.crop_width) // 2,
            self.crop_height,
            self.crop_width,
            (
                int(image.size[1] * self.resize_scale),
                int(image.size[0] * self.resize_scale),
            ),
        )

        label = Image.open(
            os.path.join(self.dataset_path, f"{self.mode}_labels", self.images[index])
        )
        label = transforms.functional.resized_crop(
            label,
            label.size[1] - self.crop_height,
            (label.size[0] - self.crop_width) // 2,
            self.crop_height,
            self.crop_width,
            (
                int(label.size[1] * self.resize_scale),
                int(label.size[0] * self.resize_scale),
            ),
        )

        if self.mode == "train" and np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)

        image = self.normalize(image).float()
        label = np.array(label)

        # One hot encode for cross entropy loss
        label = torchx.utils.encode_array(label, self.class_list).astype(np.uint8)
        label = torch.from_numpy(label).long()

        return image, label

    def normalize(self, image):
        return self.to_tensor(image)

    def denormalize(self, image):
        denormalized = image.new(*image.size())
        for i, (mean, std) in enumerate(
            zip(self.normalization_mean, self.normalization_std)
        ):
            denormalized[i, :, :] = image[i, :, :] * std + mean

        return denormalized
