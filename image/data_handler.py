from torch.utils.data import DataLoader
from img_embedding_dataset import JPEGWithEmbeddingFromChromaDataset
import multiprocessing
import chromadb


class DataHandler:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_workers = multiprocessing.cpu_count()
        self.client = chromadb.PersistentClient(path="./chromadb_storage")

    def get(self):
        train_dataset = JPEGWithEmbeddingFromChromaDataset(image_dir="openimages_custom/train",
                                                           collection_name="embeddings_train",
                                                           vector_db_client=self.client,
                                                           image_names_path="data/image_names_train.txt"
                                                           )
        test_dataset = JPEGWithEmbeddingFromChromaDataset(image_dir="openimages_custom/val",
                                                          collection_name="embeddings_val",
                                                          vector_db_client=self.client,
                                                          image_names_path="data/image_names_val.txt"
                                                          )

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
