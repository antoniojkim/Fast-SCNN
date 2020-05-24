# Fast-SCNN

Implementation of [Fast-SCNN Semantic Segmentation Architecture](https://arxiv.org/pdf/1902.04502.pdf) in PyTorch.

## Dependencies

In order to install all of the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Parameters

The parameters that are used for training and inference are to be stored in a `params.yml` file using the YaML format. There are a few required parameters as well as a number of optional parameters that have default values set. See the [training script](https://github.com/antoniojkim/Fast-SCNN/blob/master/train.py#L40) for more details.

### Data

One of the required parameters is that the data is located at `./data`. Note, if the data is located else where, recommend creating a soft link (`ln -s`) instead of moving the data. The directory structure at the data path must look as follows:

```
data/
│   class_list.csv
│
└───train/
│   │   ...
│
└───train_labels/
│   │   ...
│
└───val/
│   │   ...
│
└───val_labels/
│   │   ...
│
└───test/
│   │   ...
│
└───test_labels/
│   │   ...
```
