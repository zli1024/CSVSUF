"""
Some related functions
"""

import os
import sys
from einops import repeat, rearrange
import random
import logging
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Sampler
from ssim_torch import ssim


# Set random seed
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# Dataset
class HSVDataset(Dataset):
    """ Train dataset
    
    Args:
        root_dir (str):
            Path to directory with HSV dataset
        label (str):
            Sub-directory for each video scene
        transform (callable, optional):
            Data Preprocessing
        num_input_frames (int):
            Default: 5
            Tensor type: [B F C H W]. F = number of input frames
    """
    def __init__(self, root_dir, label, transform=True, num_input_frames=5):
        self.path = os.path.join(root_dir, label)
        self.frame_listdir = sorted(os.listdir(self.path))
        self.transform = transform
        self.num_input_frames = num_input_frames
        
    def __getitem__(self, idx):
        frames = []
        crop_size = 256
        
        # Just for H and W
        frame_name = self.frame_listdir[idx]
        frame_path = os.path.join(self.path, frame_name)
        sample = scio.loadmat(frame_path)
        sample = sample[frame_name.split('.')[0]].astype(np.float32)
        H, W, _ = sample.shape
        
        # Some random values for data augmentation
        k = random.randint(-3, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        x_idx = 0 if H == 256 else np.random.randint(0, H - crop_size)
        y_idx = np.random.randint(0, W - crop_size)
        
        for i in range(self.num_input_frames):
            frame_name = self.frame_listdir[idx+i]
            frame_path = os.path.join(self.path, frame_name)
            img = scio.loadmat(frame_path)
            img = img[frame_name.split('.')[0]].astype(np.float32) # 256 512 16 or 272 512 16
            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            if self.transform:
                img = transformData(img, x_idx, y_idx, k, vFlip, hFlip, crop_size)
            frames.append(img)
            
        # Stack for the shape of [F C H W]
        stacked_frames = torch.stack(frames)
        return stacked_frames
        
    def __len__(self):
        return len(self.frame_listdir) - self.num_input_frames + 1


class HSVDataset_val(Dataset):
    """ Validation dataset
    
    Args:
        root_dir (str):
            Path to directory with HSV dataset
        label (str):
            Sub-directory for each video scene
        transform (callable, optional):
            Data Preprocessing
        num_frames (int):
            Default: 5
            Tensor type: [B F C H W]. F = number of input frames
    """
    def __init__(self, root_dir, label, transform=None, num_input_frames=5):
        self.path = os.path.join(root_dir, label)
        self.frame_listdir = sorted(os.listdir(self.path))
        self.transform = transform
        self.num_input_frames = num_input_frames
        
    def __getitem__(self, idx):
        frames = []
        for i in range(self.num_input_frames):
            frame_name = self.frame_listdir[idx+i]
            frame_path = os.path.join(self.path, frame_name)
            img = scio.loadmat(frame_path)
            img = img[frame_name.split('.')[0]].astype(np.float32) # 256 512 16 or 272 512 16
            img = rearrange(img, 'h w c -> c h w')
            img = torch.from_numpy(img)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        
        # Stack for the shape of [F C H W]
        stacked_frames = torch.stack(frames)
        return stacked_frames
    
    def __len__(self):
        return len(self.frame_listdir) - self.num_input_frames + 1


class NonOverlappingSampler(Sampler):
    def __init__(self, data_source, num_frames_per_sample):
        self.data_source = data_source
        self.num_frames_per_sample = num_frames_per_sample
        self.valid_indices = list(range(0, len(data_source), num_frames_per_sample))

    def __iter__(self):
        return iter(self.valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    
# Data augmentation
def transformData(img, x_idx, y_idx, k, vFlip, hFlip, crop_size, nC=16):
    r""" Implementation of data augmentation, including:
    1.random crop (crop_size: 256)
    2.random rotation (degree: 90)
    3.random flip (direction: vertical, horizontal)
    """
    img_t = torch.zeros(nC, crop_size, crop_size, dtype=torch.float32)
    img_t[:, :, :] = img[:, x_idx:x_idx + crop_size, y_idx:y_idx + crop_size]
    
    # Min-Max normalization
    img_t = torch.pow(img_t, 0.5)
    for i in range(nC):
        img_t[i, :, :] = img_t[i, :, :] / torch.max(img_t[i, :, :])
        
    # Random rotation
    img_t = torch.rot90(img_t, k=k, dims=(1, 2))
    
    # Random vertical flip
    for j in range(vFlip):
        img_t = torch.flip(img_t, dims=(1,))
        
    # Random horizontal flip
    for j in range(hFlip):
        img_t = torch.flip(img_t, dims=(2,))
        
    return img_t


class normalize(torch.nn.Module):
    def forward(self, img):
        # Min-Max normalization
        img = torch.pow(img, 0.5)
        for i in range(16):
            img[i, :, :] = img[i, :, :] / torch.max(img[i, :, :])
        return img


# Masks
def init_masks(args):
    r""" Generate a batch of temporal patch of masks
    MasksBatch shape: [B F C H W] -> [B 5 16 256 256]
    """
    masks_fn = sorted(os.listdir(args['masks_path']))
    MasksBatch = torch.zeros(args['batch_size'], args['temp_patch_size'], 16, 256, 256)
    for i in range(args['temp_patch_size']):
        mask = scio.loadmat(os.path.join(args['masks_path'], masks_fn[i]))
        mask = mask[masks_fn[i].split('.')[0]].astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = repeat(mask, 'h w -> b c h w', b=args['batch_size'], c=16)
        MasksBatch[:, i, :, :, :] = mask
    return MasksBatch


# Measurements
def init_meas(gt, masks_batch, device):
    r""" Generate input measurements y
    y: [B F H W] -> [B 5 256 286]
    
    Args:
        gt: [B F C H W] -> [B 5 16 256 256]
        masks_batch: [B F C H W] -> [B 5 16 256 256]
    """
    temp_batch = torch.mul(gt, masks_batch)
    temp_shift = shift(temp_batch, device=device, step=2)
    meas_y = torch.sum(temp_shift, 2)
    return meas_y


# H and A matrix
def get_H_A(args, device):
    r""" Get H and A matrices
    """
    F, C, H, W, step = 5, 16, 256, 256, 2
    
    # H matrix (CSR format)
    H_mtx = scio.loadmat(os.path.join(args['data_root'], 'H.mat'))
    H_mtx = H_mtx['H'].tocsr()
    H_ptr = torch.tensor(H_mtx.indptr, dtype=torch.long)
    H_ind = torch.tensor(H_mtx.indices, dtype=torch.long)
    H_val = torch.tensor(H_mtx.data, dtype=torch.float32)
    H_mtx = torch.sparse_csr_tensor(H_ptr, H_ind, H_val, H_mtx.shape)
    assert H_mtx.shape == (H*(W+step*(C-1))*F, H*W*C*F), f"H_mtx shape mismatch! Expected {(H*(W+step*(C-1))*F, H*W*C*F)}, but got {H_mtx.shape}"

    # A matrix (CSR format)
    A_mtx = scio.loadmat(os.path.join(args['data_root'], 'A.mat'))
    A_mtx = A_mtx['A'].tocsr()
    A_ptr = torch.tensor(A_mtx.indptr, dtype=torch.long)
    A_ind = torch.tensor(A_mtx.indices, dtype=torch.long)
    A_val = torch.tensor(A_mtx.data, dtype=torch.float32)
    A_mtx = torch.sparse_csr_tensor(A_ptr, A_ind, A_val, A_mtx.shape)
    assert A_mtx.shape == (H*W*C*F, H*(W+step*(C-1))*F), f"A_mtx shape mismatch! Expected {(H*W*C*F, H*(W+step*(C-1))*F)}, but got {A_mtx.shape}"

    return H_mtx.to(device), A_mtx.to(device)


# Shift operation
def shift(inputs, device, step=2):
    r""" Simulation of dispersive effects
    Shifting step sets to 2
    output: [B F H W] -> [B 5 256 286]
    
    Args:
        input: [B F C H W] -> [B 5 16 256 256]
        step (int): 2
    """
    [B, F, C, H, W] = inputs.shape
    output = torch.zeros(B, F, C, H, W + step * (C - 1)).to(device).float()
    for i in range(C):
        output[:, :, i, :, (i * step):(i * step + W)] = inputs[:, :, i, :, :]
    return output


# PSNR computation
def comp_psnr(img, ref): # input: [B F C H W]
    img = torch.clamp((img*256).round(), 0, 255)
    ref = torch.clamp((ref*256).round(), 0, 255)
    img_cat = rearrange(img, 'b f c h w -> (b f c) h w')
    ref_cat = rearrange(ref, 'b f c h w -> (b f c) h w')
    nC = img_cat.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img_cat[i, :, :] - ref_cat[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 ** 2) / mse)
    return psnr / nC


# SSIM computation
def comp_ssim(img, ref): # input: [B F C H W]
    B, F, _, _, _ = img.shape
    ssim_list = []
    img = rearrange(img, 'b f c h w -> (b f) c h w')
    ref = rearrange(ref, 'b f c h w -> (b f) c h w')
    for i in range(B*F):
        ssim_v = ssim(torch.unsqueeze(img[i, :, :, :], 0), torch.unsqueeze(ref[i, :, :, :], 0))
        ssim_list.append(ssim_v.detach().cpu().numpy())
    return np.mean(np.asarray(ssim_list))


# time logging
def time2filename(time):
    r""" Format conversion to file name logging
    """
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


# file logging
def init_logger(args, log_path):
    r""" Return summarywriter and file logger
    """
    # Writer
    writer = SummaryWriter(log_path)
    
    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    
    #-- Formatter
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s %(message)s', 
                                  datefmt='%Y/%m/%d %H:%M:%S')
    
    #-- Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    #-- File Handler
    file_handler = logging.FileHandler(os.path.join(log_path, 'train.log'), mode='a+')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return writer, logger

 
if __name__ == '__main__':
    val = torch.ones((5), dtype=torch.float32)
    row_ind = torch.arange(5, dtype=torch.int64)
    col_ptr = torch.arange(5+1, dtype=torch.int64)
    I = torch.sparse_csc_tensor(col_ptr, row_ind, val, (5, 5))
