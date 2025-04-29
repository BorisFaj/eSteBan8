from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import chromadb
import os

class JPEGWithEmbeddingFromChromaDataset(Dataset):
    def __init__(self, image_dir, collection_name, image_size=224):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith('.jpg')
        ])

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.client = chromadb.PersistentClient(path="./chromadb_storage")
        self.collection = self.client.get_collection(name=collection_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).float()

        image_id = os.path.splitext(os.path.basename(image_path))[0]

        query = self.collection.get(ids=[image_id])
        embedding = torch.tensor(query["embeddings"][0]).float()

        return image, embedding
