#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Model.FastSCNN import *
from Dataset.dataset import *


# In[2]:


import os
import time

from tqdm import tqdm
import matplotlib.pyplot as plt

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


# # Parameters
# 
# The parameters are read from a [yaml](https://en.wikipedia.org/wiki/YAML) file.

# In[3]:


with open("params.yml") as file:
    params = load(file, Loader=Loader)


# In[4]:


# required
dataset_path = params["dataset_path"]   # Path at which the dataset is located
crop_height  = params["crop_height"]    # Height of cropped/resized input image
crop_width   = params["crop_width"]     # Width of cropped/resized input image
num_classes  = params["num_classes"]    # Number of classes

# optional
num_epochs            = params.get("num_epochs", 100)                   # Number of epochs to train for
epoch_start           = params.get("epoch_start", 0)                    # Start counting epochs from this number
batch_size            = params.get("batch_size", 4)                     # Number of images in each batch
checkpoint_step       = params.get("checkpoint_step", 2)                # How often to save checkpoints (epochs)
validation_step       = params.get("validation_step", 2)                # How often to perform validation (epochs)
num_validation        = params.get("num_validation", 1000)              # How many validation images to use
num_workers           = params.get("num_workers", 4)                    # Number of workers
learning_rate         = params.get("learning_rate", 0.045)              # learning rate used for training
cuda                  = params.get("cuda", "0,1")                       # GPU ids used for training  
use_gpu               = params.get("use_gpu", True)                     # whether to user gpu for training
pretrained_model_path = params.get("pretrained_model_path", None)       # path to pretrained model
save_model_path       = params.get("save_model_path", "./checkpoints")  # path to save model
log_file              = params.get("log_file", "./log.txt")             # path to log file

use_gpu = use_gpu and torch.cuda.is_available()


# In[5]:


if crop_height * 2 != crop_width:
    raise AssertionError("Crop width must be exactly twice the size of crop height")


# # Dataset

# In[6]:


# Check to see if all required paths are present
if not os.path.join(dataset_path, "class_dict.csv"):
    raise AssertionError(os.path.join(dataset_path, "class_dict.csv") + " does not exist")

for directory in ("train", "train_labels", "test", "test_labels", "val", "val_labels"):
    if not os.path.isdir(os.path.join(dataset_path, directory)):
        raise AssertionError(os.path.join(dataset_path, directory) + " does not exist")


# ## Training Dataset

# In[7]:


training_dataset = Dataset(dataset_path, crop_height, crop_width, mode="train")
training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size  = batch_size,
    num_workers = num_workers,
    shuffle     = True,
    drop_last   = True
)


# ## Validation Dataset

# In[8]:


val_dataset = Dataset(dataset_path, crop_height, crop_width, mode="val")
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size  = 1,
    num_workers = num_workers,
    shuffle     = True
)

num_validation = min(num_validation, len(val_dataloader))


# # Train

# In[9]:


os.environ['CUDA_VISIBLE_DEVICES'] = cuda


# In[10]:


model = FastSCNN(image_height   = crop_height,
                 image_width    = crop_width,
                 image_channels = 3,
                 num_classes    = num_classes)


# In[11]:


if use_gpu:
    model = torch.nn.DataParallel(model).cuda()


# In[12]:


num_parameters = sum(p.numel() for p in model.parameters())
num_parameters


# In[13]:


# load pretrained model if exists
if pretrained_model_path is not None:
    print('loading model from %s ...' % pretrained_model_path)
    model.module.load_state_dict(torch.load(pretrained_model_path))


# ## Validation

# In[14]:


reverse_one_hot = lambda image: torch.argmax(image.permute(1, 2, 0), dim=-1)

def compute_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

iou_per_class = lambda hist: (np.diag(hist) + 1e-5) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-5)

def run_validation(epoch):

    with torch.no_grad(), tqdm(total=num_validation, position=0, leave=False) as val_progress:
        model.eval()
    
        val_progress.set_description('validation epoch %d' % epoch)
        
        precisions = []
        hist = np.zeros((num_classes, num_classes))

        for i, (data, label) in enumerate(val_dataloader):
            
            if use_gpu:
                data = data.cuda()
#                 label = label.cuda()
                
            output = model(data).squeeze()
            output = reverse_one_hot(output)
            
            if use_gpu:
                output = np.array(output.detach().cpu())
            else:
                output = np.array(output)

            label = label.squeeze()
            label = np.array(label)

            precisions.append(np.sum(output == label) / np.prod(label.shape))
            hist += compute_hist(output.flatten(), label.flatten(), num_classes)
            
            val_progress.set_postfix(precision='%.6f' % np.mean(precisions))
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

# In[15]:


optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer.zero_grad()


# In[16]:


criterion = torch.nn.CrossEntropyLoss()


# In[17]:


class poly_lr_scheduler:
    
    def __init__(self, optimizer, base_lr, epochs, niters_per_epoch = 1, power=0.9):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.curr_lr = base_lr
        self.power = power
        self.T = 0
        self.N = epochs * niters_per_epoch
        
    def __call__(self):
        self.T = min(self.T + 1, self.N)
        self.curr_lr = self.base_lr * (1 - self.T / (self.N - 1)) ** self.power
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.curr_lr
        

lr_scheduler = poly_lr_scheduler(optimizer,
                                 base_lr = learning_rate,
                                 epochs = num_epochs,
                                 niters_per_epoch = len(training_dataloader))


# In[18]:


with open(log_file, 'w') as log:
    
    def write_log(msg):
        log.write(msg)
        log.write(os.linesep)
        
    write_log("time,{time}".format(time=time.time()))
    write_log(",epoch,loss")

    
    max_miou = 0
    step = 0
    for epoch in range(epoch_start, num_epochs):
        
        with tqdm(total=len(training_dataloader), position=0, leave=False) as progress:
            
            progress.set_description('epoch %d, lr %f' % (epoch, lr_scheduler.curr_lr))

            for i, (images, labels) in enumerate(training_dataloader):
                lr_scheduler()  # Updated learning rate

                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                output = model(images)
                loss = criterion(output, labels)

                progress.set_postfix(loss='%.6f' % loss)
                progress.update()

                loss.backward()

                if i % 3 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.set_description('epoch %d, lr %f' % (epoch, lr_scheduler.curr_lr))

                write_log("{step},{epoch},{loss}".format(step=step, epoch=epoch, loss=loss.item()))
                step += 1



            if epoch > 0 and epoch % checkpoint_step == 0:
                if not os.path.isdir(save_model_path):
                    os.mkdir(save_model_path)

                torch.save(model.module.state_dict(),
                           os.path.join(save_model_path, 'latest_model.pt'))
            
            
        write_log("time,{time}".format(time=time.time()))
            
        if epoch % validation_step == 0:
            
            precision, miou, iou = run_validation(epoch)
            if miou > max_miou:
                max_miou = miou
                
                if not os.path.isdir(save_model_path):
                    os.mkdir(save_model_path)
                    
                torch.save(model.module.state_dict(),
                           os.path.join(save_model_path, 'best_model.pt'))
                
            write_log("precision,{epoch},{precision}".format(epoch=epoch, precision=precision))
            write_log("miou,{epoch},{miou}".format(epoch=epoch, miou=miou))
            write_log("iou,{epoch},{iou}".format(epoch=epoch, iou=iou))
            write_log("time,{time}".format(time=time.time()))       

