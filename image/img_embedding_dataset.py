from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import os

class JPEGWithEmbeddingFromChromaDataset(Dataset):
    def __init__(self, image_dir, collection_name, vector_db_client, image_size=224, batch_size=100):
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

        self.embeddings = {}
        for i in range(0, len(self.image_names), batch_size):
            batch_ids = self.image_names[i:i + batch_size]
            results = self.collection.get(ids=batch_ids, include=["embeddings"])

            if not results:
                continue

            ids = results.get("ids", [])
            embeddings = results.get("embeddings", [])

            if len(ids) > 0 and len(embeddings) > 0:
                for id_, emb in zip(ids, embeddings):
                    if emb is not None:
                        self.embeddings[id_] = torch.tensor(emb, dtype=torch.float32)

        print(f"✅ Precargados {len(self.embeddings)} embeddings de {len(self.image_names)} imágenes.")

        valid_pairs = [
            (path, name)
            for path, name in zip(self.image_paths, self.image_names)
            if name in self.embeddings
        ]

        if not valid_pairs:
            raise ValueError("🚫 No se ha encontrado ninguna imagen con embedding válido.")

        self.image_paths, self.image_names = zip(*valid_pairs)

        print(f"📦 Dataset final contiene {len(self.image_paths)} imágenes con embeddings.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        embedding = self.embeddings[image_name]
        return image, embedding
