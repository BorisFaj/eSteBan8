from PIL import Image, ImageFile
from torchvision import transforms
import os
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WorldToFaceDataset(torch.utils.data.Dataset):
    def __init__(self, faces_dir, no_faces_dir, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        # 👇 Solo inputs sin caras
        self.inputs = [
            os.path.join(no_faces_dir, f)
            for f in os.listdir(no_faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # 👇 Solo objetivos (caras reales)
        self.real_faces = [
            os.path.join(faces_dir, f)
            for f in os.listdir(faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not self.inputs:
            raise ValueError("❌ No se encontraron imágenes en no_faces_dir")

        if not self.real_faces:
            raise ValueError("❌ No se encontraron imágenes en faces_dir")

        self.image_size = image_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_path = self.inputs[idx]
        real_face_path = random.choice(self.real_faces)

        input_img = Image.open(input_path).convert("RGB")
        real_img = Image.open(real_face_path).convert("RGB")

        input_tensor = self.transform(input_img)
        real_tensor = self.transform(real_img)

        return input_tensor, real_tensor
