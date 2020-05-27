from FastSCNN import Dataset
from multiprocessor import Multiprocessor
from PIL import Image

for mode in ("train", "val", "test"):
    dataset = Dataset(960, 1920, 0.5, mode)

    def preprocess(indices):
        for i in indices:
            image, label = dataset[i]
            image.numpy().tofile(f"data/roadline/{mode}/{dataset.images[i]}")
            label.numpy().tofile(f"data/roadline/{mode}_labels/{dataset.images[i]}")

    processor = Multiprocessor(cpus=8)
    processor.process(preprocess, list(range(len(dataset))))



