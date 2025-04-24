from torch.utils.data import DataLoader
from img_embedding_dataset import JPEGAndEmbeddingDataset

class DataHandler:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def get(self):
        train_dataset = JPEGAndEmbeddingDataset(image_dir="openimages_custom/train/default", embedding_dir="data/train_preprocessed")
        test_dataset = JPEGAndEmbeddingDataset(image_dir="openimages_custom/val/default", embedding_dir="data/val_preprocessed")

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return train_dataset, train_loader, test_dataset, test_loader
