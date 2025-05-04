from torchvision import transforms
from PIL import Image
import os
import random
import torch

class WorldToFaceDataset(torch.utils.data.Dataset):
    def __init__(self, faces_dir, no_faces_dir, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.inputs = []
        self.real_faces = []

        for path in os.listdir(faces_dir):
            full_path = os.path.join(faces_dir, path)
            self.inputs.append(full_path)
            self.real_faces.append(full_path)  # también serán nuestras imágenes reales

        for path in os.listdir(no_faces_dir):
            full_path = os.path.join(no_faces_dir, path)
            self.inputs.append(full_path)

        random.shuffle(self.inputs)
        self.image_size = image_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_path = self.inputs[idx]
        input_img = Image.open(input_path).convert("RGB")
        input_tensor = self.transform(input_img)

        # Imagen real con cara aleatoria (solo de la carpeta de caras)
        real_face_path = random.choice(self.real_faces)
        real_img = Image.open(real_face_path).convert("RGB")
        real_tensor = self.transform(real_img)

        return input_tensor, real_tensor
