#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:


import os

import numpy as np
import torch
import torchx
from tqdm import tqdm, trange

from FastSCNN import AudiDataset, FastSCNN


# # Parameters

# In[2]:


params = torchx.params.Parameters("params.yml")


# In[3]:


if params.crop_height * 2 != params.crop_width:
    raise AssertionError("Crop width must be exactly twice the size of crop height")


# In[4]:


log = params.get_logger("FastSCNN")


# # Dataset

# In[5]:

# Check to see if all required paths are present
if not os.path.join("data", "class_list.json"):
    raise AssertionError(os.path.join("data", "class_list.json") + " does not exist")

for directory in ("train", "train_labels", "test", "test_labels", "val", "val_labels"):
    if not os.path.isdir(os.path.join("data", directory)):
        raise AssertionError(os.path.join("data", directory) + " does not exist")


# In[6]:


num_classes = 2 if params.single_class else len(AudiDataset.classes) + 1


# ## Training Dataset

# In[7]:


train_dataloader = torch.utils.data.DataLoader(
    AudiDataset(
        params.crop_height,
        params.crop_width,
        params.resize_scale,
        mode="train",
        single_class=params.single_class,
    ),
    batch_size=params.batch_size,
    num_workers=params.num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)


# ## Validation Dataset

# In[8]:


val_dataset = AudiDataset(
    params.crop_height,
    params.crop_width,
    params.resize_scale,
    mode="val",
    single_class=params.single_class,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, num_workers=params.num_workers, shuffle=True, pin_memory=True,
)

num_validation = min(params.num_validation, len(val_dataloader))


# # Train

# In[9]:


device = torch.device("cuda:" + params.cuda if params.use_gpu else "cpu")


def to_device(tensor):
    if params.use_gpu:
        # torch.nn.DataParallel(model).cuda()
        return  tensor.cuda()
        # return tensor.to(device)
    else:
        return tensor


# In[10]:


model = FastSCNN(
    image_height=int(params.crop_height * params.resize_scale),
    image_width=int(params.crop_width * params.resize_scale),
    image_channels=3,
    num_classes=num_classes,
)
model = to_device(model)


# In[11]:


print("Number of Trainable Parameters: ", model.num_params())


# In[12]:


# load pretrained model if exists
if params.pretrained_model_path is not None:
    print("loading model from %s ..." % params.pretrained_model_path)
    try:
        model.load(params.pretrained_model_path)
    except RuntimeError:
        pass

model.save(os.path.join(params.save_model_path, f"latest_{params.model_suffix}.pt"))

# ## Validation

def compute_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def iou_per_class(hist):
    return (np.diag(hist) + 1e-5) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-5)


def validation(model, dataloader):

    with torch.no_grad(), tqdm(
        total=num_validation, position=0, leave=False
    ) as val_progress:
        model.eval()

        precisions = []
        hist = np.zeros((num_classes, num_classes))

        for i, (data, label) in enumerate(dataloader):

            data = to_device(data)

            output = model(data).argmax(dim=1).detach().cpu().numpy()
            label = label.numpy()

            precisions.append(np.sum(output == label) / np.prod(label.shape))
            hist += compute_hist(output.flatten(), label.flatten(), num_classes)

            val_progress.set_postfix(precision="%.6f" % np.mean(precisions))
            val_progress.update()

            if i >= num_validation:
                break

            if len(precisions) > num_validation:
                raise AssertionError("Validating more than wanted")

        precision = np.mean(precisions)
        iou = iou_per_class(hist)[:-1]
        miou = np.mean(iou)

    return precision, miou, iou


# ## Training Loop

# In[16]:


optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4
)
criterion = torch.nn.CrossEntropyLoss()


# In[ ]:


model_suffix = ""

best_mean_loss = np.inf
if params.pretrained_model_path is not None:
    best_mIoU, best_precision, iou = validation(model, val_dataloader)
    log.info(f"Initial: mIoU {best_mIoU}, precision {best_precision}")
else:
    best_mIoU = 0
    best_precision = 0

with torchx.optim.lr_scheduler.PolynomialLR(
    optimizer, power=4, max_decay_steps=50, final_learning_rate=0.00005
) as scheduler:

    for epoch in trange(params.epoch_start, params.num_epochs + 1, leave=False):

        with tqdm(
            total=len(train_dataloader), position=0, leave=False
        ) as progress, torchx.batch.BatchAccumulator(optimizer, 1) as batchAccumulator:
            model.train()

            progress.set_description(
                "epoch %d, lr %f" % (epoch, scheduler.learning_rate)
            )

            cumulative_loss = 0
            for i, (images, labels) in enumerate(train_dataloader):

                images = to_device(images)
                labels = to_device(labels)

                output = model.forward(images).float()
                loss = criterion(output, labels)
                cumulative_loss += loss.item()

                info = f"Epoch {epoch}, lr {scheduler.learning_rate}, i {i}, loss {loss.item()}"  # noqa: E501
                if np.isnan(loss.item()):
                    log.warning(info)
                    break
                else:
                    log.info(info)

                loss.backward()

                progress.set_postfix(loss="%.6f" % loss.item())
                progress.update()

                next(batchAccumulator)

            if epoch % params.checkpoint_step == 0:
                model.save(
                    os.path.join(
                        params.save_model_path, f"latest_{params.model_suffix}.pt"
                    )
                )

            mean_loss = cumulative_loss / len(train_dataloader)
            log.info(f"Epoch {epoch}, mean loss {mean_loss}")

            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                model.save(
                    os.path.join(
                        params.save_model_path, f"min_loss_{params.model_suffix}.pt"
                    )
                )

            if epoch % params.validation_step == 0:
                mIoU, precision, iou = validation(model, val_dataloader)
                log.info(f"Epoch {epoch}, mIoU {mIoU}, precision {precision}")
                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    model.save(
                        os.path.join(
                            params.save_model_path,
                            f"best_miou_{params.model_suffix}.pt",
                        )
                    )
                if precision > best_precision:
                    best_precision = precision
                    model.save(
                        os.path.join(
                            params.save_model_path,
                            f"best_precision_{params.model_suffix}.pt",
                        )
                    )
