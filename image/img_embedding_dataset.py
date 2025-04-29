from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import os

class JPEGWithEmbeddingFromChromaDataset(Dataset):
    def __init__(self, image_dir, collection_name, vector_db_client, image_size=224, preload_batch_size=2048):
        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.image_paths = [os.path.join(image_dir, name) for name in self.image_names]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.collection = vector_db_client.get_collection(name=collection_name)
        self.preload_batch_size = preload_batch_size

        self.cached_embeddings = {}
        self.cached_indices = []

    def preload_embeddings(self, start_idx):
        # Cargar embeddings en caché para un bloque de imágenes
        end_idx = min(start_idx + self.preload_batch_size, len(self.image_paths))
        batch_names = self.image_names[start_idx:end_idx]

        results = self.collection.get(ids=batch_names, include=["embeddings"])
        embeddings = results.get("embeddings", [])
        ids = results.get("ids", [])

        self.cached_embeddings = {}
        for id_, emb in zip(ids, embeddings):
            if emb is not None:
                self.cached_embeddings[id_] = torch.tensor(emb, dtype=torch.float16)

        self.cached_indices = list(range(start_idx, end_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx not in self.cached_indices:
            # Cargar nuevo bloque de embeddings si idx fuera de caché
            block_start = (idx // self.preload_batch_size) * self.preload_batch_size
            self.preload_embeddings(block_start)

        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        embedding = self.cached_embeddings.get(image_name)

        if embedding is None:
            raise ValueError(f"🚫 No se encontró embedding para {image_name}")

        return image, embedding
