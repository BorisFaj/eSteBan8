from torch.utils.data import DataLoader
from img_embedding_dataset import JPEGWithEmbeddingFromChromaDataset
import multiprocessing

class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = multiprocessing.cpu_count()

    def get(self):
        train_dataset = JPEGWithEmbeddingFromChromaDataset(image_dir="openimages_custom/train/default", collection_name="embeddings_train")
        test_dataset = JPEGWithEmbeddingFromChromaDataset(image_dir="openimages_custom/val/default", collection_name="embeddings_train")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

        return train_dataset, train_loader, test_dataset, test_loader
