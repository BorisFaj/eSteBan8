from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch
import os

class JPEGAndEmbeddingDataset(Dataset):
    def __init__(self, image_dir, embedding_dir, image_size=224):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith('.jpg')
        ])
        self.embedding_paths = sorted([
            os.path.join(embedding_dir, f)
            for f in os.listdir(embedding_dir) if f.endswith('.pt')
        ])

        assert len(self.image_paths) == len(self.embedding_paths), \
            f"Desajuste: {len(self.image_paths)} imágenes vs {len(self.embedding_paths)} embeddings"

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image).half()  # float16

        embedding = torch.load(self.embedding_paths[idx], map_location="cpu")  # ya está en float16

        return image, embedding
