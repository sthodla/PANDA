import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

class Panda(Dataset):
    def __init__(self, df, transforms=None, image_path='/mnt/wamri/panda/train_images/', 
                 mask_path='/mnt/wamri/panda/train_label_masks'):
        '''
        df: A csv file containing info of the dataset
        transforms: Albumentations transforms
        image_path: path to images
        mask_path: path to masks
        '''
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.image_path = image_path
        self.mask_path = mask_path
    
    def __len__(self): return len(self.df)
    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx, :].values
        image_id, provider, isup = row[0], row[1], row[2]
        img = tiff.imread(os.path.join(self.image_path, f'{image_id}.tiff'))
        mask = os.path.join(self.mask_path, f'{image_id}_mask.tiff')
        # Not all images have a mask
        if os.path.exists(mask):
            mask = tiff.imread(mask)
        else:
            mask = np.zeros_like(img)
        mask = mask[:,:,0]
        
        # Data Augmentation
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
            
        return {'image': torch.Tensor(img).permute(2, 0, 1), 'mask': torch.Tensor(mask), 
                'isup': torch.Tensor([isup]), 'id': image_id, 'prov': provider}