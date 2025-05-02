import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageDataset(Dataset):
    def __init__(self, real_dir, fake_dir, image_size=224):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.image_size = image_size

        self.real_img_paths = sorted([
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.fake_img_paths = sorted([
            os.path.join(fake_dir, f)
            for f in os.listdir(fake_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        # Recortar hasta el mínimo común
        min_len = min(len(self.real_img_paths), len(self.fake_img_paths))
        self.real_img_paths = self.real_img_paths[:min_len]
        self.fake_img_paths = self.fake_img_paths[:min_len]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.real_img_paths)

    def __getitem__(self, idx):
        real_img = Image.open(self.real_img_paths[idx]).convert("RGB")
        fake_img = Image.open(self.fake_img_paths[idx]).convert("RGB")
        return self.transform(real_img), self.transform(fake_img)
