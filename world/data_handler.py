from torchvision import transforms
from PIL import Image
import os
import random
import torch

class WorldToFaceDataset(torch.utils.data.Dataset):
    def __init__(self, faces_dir, no_faces_dir, real_faces_dir, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.input_samples = []

        for path in os.listdir(faces_dir):
            self.input_samples.append(os.path.join(faces_dir, path))

        for path in os.listdir(no_faces_dir):
            self.input_samples.append(os.path.join(no_faces_dir, path))

        self.real_faces = sorted([
            os.path.join(real_faces_dir, path)
            for path in os.listdir(real_faces_dir)
        ])

        self.image_size = image_size
        random.shuffle(self.input_samples)

    def __len__(self):
        return len(self.input_samples)

    def __getitem__(self, idx):
        # Imagen de entrada (puede tener cara o no)
        input_path = self.input_samples[idx]
        input_img = Image.open(input_path).convert("RGB")
        input_tensor = self.transform(input_img)

        # Imagen real de cara (sample aleatorio)
        real_face_path = random.choice(self.real_faces)
        real_img = Image.open(real_face_path).convert("RGB")
        real_tensor = self.transform(real_img)

        return input_tensor, real_tensor
