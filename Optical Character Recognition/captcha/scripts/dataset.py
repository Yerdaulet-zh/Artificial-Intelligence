import torch

from PIL import Image

from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, image_paths, targets, transformer=None):
        super(dataset, self).__init__()
        self.targets = targets
        self.image_paths = image_paths
        self.transformer = transformer
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, x):
        image = Image.open(fr"{self.image_paths[x]}").convert("RGB")
        target = self.targets[x]
        
        if self.transformer:
            image = self.transformer(image)
        image = image.to(torch.float)
        target = torch.tensor(target, dtype=torch.long)
        return image, target
    