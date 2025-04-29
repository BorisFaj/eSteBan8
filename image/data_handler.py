from torch.utils.data import DataLoader
from img_embedding_dataset import JPEGWithEmbeddingFromChromaDataset
import multiprocessing
import chromadb

class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # Reservamos 1-2 cores para el sistema, no todos
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)
        self.client = chromadb.PersistentClient(path="./chromadb_storage")

    def get(self):
        train_dataset = JPEGWithEmbeddingFromChromaDataset(
            image_dir="openimages_custom/train",
            collection_name="embeddings_train",
            vector_db_client=self.client
        )

        test_dataset = JPEGWithEmbeddingFromChromaDataset(
            image_dir="openimages_custom/val",
            collection_name="embeddings_val",
            vector_db_client=self.client
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),  # Menos carga en validación
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        return train_dataset, train_loader, test_dataset, test_loader
