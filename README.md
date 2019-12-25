# Fast-SCNN

Implementation of Fast-SCNN Semantic Segmentation Architecture in PyTorch.

## Dependencies

In order to install all of the dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Parameters

The parameters that are used for training and inference are to be stored in a `params.yml` file using the YaML format. There are a few required parameters as well as a number of optional parameters that have default values set. See the [training script](https://github.com/antoniojkim/Fast-SCNN/blob/master/train.py#L40) for more details.

### File Structure

One of the required parameters is the path to the dataset. The file structure at this path must look like the following:

```
path/to/dataset
│   class_dict.csv
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
