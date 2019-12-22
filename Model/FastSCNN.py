#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F


# In[2]:


class DSConv(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(DSConv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# In[3]:


def Conv2dBatch(in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels  = in_channels,
                        out_channels = out_channels,
                        kernel_size  = kernel_size,
                        stride       = stride,
                        padding      = padding,
                        bias         = False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )

def DSConv(in_channels, out_channels, stride=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )

def DWConv(in_channels, out_channels, stride=1, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True)
    )


# In[4]:


class Bottleneck(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(Bottleneck, self).__init__()
        
        self.shortcut = stride == 1 and in_channels == out_channels
        self.block = torch.nn.Sequential(
            Conv2dBatch(in_channels  = in_channels,
                        out_channels = in_channels * t,
                        kernel_size  = 1,
                        stride       = 1),
            DWConv(in_channels  = in_channels * t,
                   out_channels = in_channels * t,
                   stride       = stride),
            torch.nn.Conv2d(in_channels  = in_channels * t,
                            out_channels = out_channels,
                            kernel_size  = 1,
                            bias         = False),
            torch.nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        if self.shortcut:
            return x + self.block(x)
        
        else:
            return self.block(x)


# In[5]:


class PyramidPooling(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = Conv2dBatch(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = Conv2dBatch(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = Conv2dBatch(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = Conv2dBatch(in_channels, inter_channels, 1, **kwargs)
        self.out   = Conv2dBatch(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = torch.nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='nearest') # mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        
        return x


# In[6]:


class FeatureFusionModule(torch.nn.Module):

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = torch.nn.Sequential(
            torch.nn.Conv2d(highter_in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='nearest') # mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


# In[7]:


class FastSCNN(torch.nn.Module):
    def __init__(self, image_height=1024, image_width=2048, image_channels=3, num_classes=10, **kwargs):
        super(FastSCNN, self).__init__()
        
        self.learning_to_downsample = torch.nn.Sequential(
            Conv2dBatch(in_channels  = image_channels,
                        out_channels = 32,
                        kernel_size  = 3,
                        stride       = 2),
            DSConv(in_channels  = 32,
                   out_channels = 48,
                   stride       = 2),
            DSConv(in_channels  = 48,
                   out_channels = 64,
                   stride       = 2)
        )
        
        self.global_feature_extractor = torch.nn.Sequential(*[
                Bottleneck(in_channels  = in_channel,
                           out_channels = out_channel,
                           stride       = stride)
                for in_channel, out_channel, stride in zip(
                    (64, 64, 64,    64, 96, 96,     96, 128, 128),
                    (64, 64, 64,    96, 96, 96,    128, 128, 128),
                    ( 2,  1,  1,     2,  1,  1,      1,   1,   1)
                )
            ],
            PyramidPooling(128, 128)
        )

        self.feature_fusion_module = FeatureFusionModule(64, 128, 128)
        
        self.classifier = torch.nn.Sequential(*[
                DSConv(in_channels  = 128,
                       out_channels = 128,
                       kernel_size  = 1)
                for n in range(2)
            ],
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(in_channels  = 128,
                            out_channels = num_classes,
                            kernel_size  = 1)
        )
        
        
    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion_module(higher_res_features, x)
        x = self.classifier(x)
        
#         outputs = []
        x = F.interpolate(x, size, mode='nearest') # mode='bilinear', align_corners=True)
#         outputs.append(x)
        
        return x


# In[8]:


# img = torch.randn(1, 3, 960, 1920)
# model = FastSCNN(image_height=960, image_width=1920, image_channels=3)


# In[10]:


# %%timeit

# with torch.no_grad():
#     model.eval()
    
#     label = model(img)

