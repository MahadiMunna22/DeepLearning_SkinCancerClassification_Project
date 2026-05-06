from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random


class HAM10000FullDataset(Dataset):
    def __init__(self, image_dir, mask_dir, df, image_size=224):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.df = df.set_index("image_id")

        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples = []
        for img_path in self.image_dir.glob("*.jpg"):
            image_id = img_path.stem
            mask_path = self.mask_dir / f"{image_id}_segmentation.png"
            if image_id in self.df.index and mask_path.exists():
                label = self.df.loc[image_id]["label"]
                self.samples.append((img_path, mask_path, label, image_id))

        print(f"Total valid samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def get_labels(self):
        return [int(label) for _, _, label, _ in self.samples]

    def __getitem__(self, idx):
        img_path, mask_path, label, image_id = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask) / 255.0
        masked_image_np = image_np * mask_np[:, :, np.newaxis]
        masked_image_pil = Image.fromarray(masked_image_np.astype(np.uint8))

        image = self.image_transform(image)
        masked_image = self.image_transform(masked_image_pil)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return {
            "image": image,           # [3, H, W]
            "mask": mask,             # [1, H, W]
            "masked_image": masked_image,  # [3, H, W]
            "label": torch.tensor(label),
            "image_id": image_id,
        } 


class HAM10000AugmentedDataset(Dataset):
    def __init__(self, image_dir, mask_dir, df, image_size=224):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.df = df.set_index("image_id")
        self.image_size = image_size

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((image_size, image_size))
        self.resize_nearest = T.Resize(
            (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
        )

        self.color_jitter = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

        self.samples = []
        for img_path in self.image_dir.glob("*.jpg"):
            image_id = img_path.stem
            mask_path = self.mask_dir / f"{image_id}_segmentation.png"
            if image_id in self.df.index and mask_path.exists():
                label = self.df.loc[image_id]["label"]
                self.samples.append((img_path, mask_path, label, image_id))

        print(f"Total valid samples (augmented): {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def get_labels(self):
        return [int(label) for _, _, label, _ in self.samples]

    def _joint_augment(self, image: Image.Image, mask: Image.Image):
        """Apply the same geometric transforms to both image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random 90° rotation (0, 90, 180, 270)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random small free rotation ±30°
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Random crop & resize
        if random.random() > 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size))
            mask = TF.resized_crop(
                mask, i, j, h, w, (self.image_size, self.image_size),
                interpolation=TF.InterpolationMode.NEAREST
            )

        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path, label, image_id = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Joint geometric augmentation (image + mask stay aligned)
        image, mask = self._joint_augment(image, mask)

        # Color jitter — image only
        image = self.color_jitter(image)

        # Build masked image after all augments
        image_np = np.array(image)
        mask_np = np.array(mask) / 255.0
        masked_image_np = image_np * mask_np[:, :, np.newaxis]
        masked_image_pil = Image.fromarray(masked_image_np.astype(np.uint8))

        # Resize + to tensor
        image = self.to_tensor(self.resize(image))
        masked_image = self.to_tensor(self.resize(masked_image_pil))
        mask = self.to_tensor(self.resize_nearest(mask))
        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "label": torch.tensor(label),
            "image_id": image_id,
        }