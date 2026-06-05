import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class UdacityDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Read the CSV and explicitly state there is no header row
        self.driving_log = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        
        # Standard PilotNet Transformation
        self.transform = transforms.Compose([
            # 1. Crop out the sky and the car hood (Focus only on the road)
            # Udacity raw imgs are 320x160. Crop top 60px (sky) and bottom 25px (hood)
            transforms.Lambda(lambda img: img.crop((0, 60, 320, 135))), 
            
            # 2. Resize to PilotNet's native architecture input
            transforms.Resize((37, 117)),
            
            # 3. Convert to Tensor (scales pixels from 0-255 to 0.0-1.0)
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.driving_log)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the raw string from the CSV (column 0 is center image)
        raw_path = self.driving_log.iloc[idx, 0].strip()
        
        # Replace Windows backslashes with forward slashes, then split to get the filename
        img_name = raw_path.replace('\\', '/').split('/')[-1]
        
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get the steering angle (Column 3)
        steering_angle = float(self.driving_log.iloc[idx, 3])
        
        return image, torch.tensor([steering_angle], dtype=torch.float32)
