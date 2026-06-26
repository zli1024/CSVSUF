import os
import datetime
import time
import tempfile
import math
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler
from torchvision.transforms import v2 
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis

from utils import *
from options import opt
from architectures.CSVSUF import CSVSUF

from distributed_utils import init_distributed_mode, dist, cleanup, reduce_value, is_main_process
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def main(**args):
    r""" Main training
    """
    # -----> Sec 0 Preprocessing <-----
    # Set seed
    set_seed(args['seed'])
    
    # Define devices
    if torch.cuda.is_available() is False:
        raise EnvironmentError("No GPU available!")
    
    init_distributed_mode(args=opt)
    
    rank = opt.rank
    device = torch.device(opt.device)
    checkpoint_path = ""
    
    # Saving path
    if rank == 0:
        date_time = str(datetime.datetime.now())
        date_time = time2filename(date_time)
        model_path = args['outf'] + "/train/" + date_time + "/model_pth/"
        log_path = args['outf'] + "/train/" + date_time + "/logs/"
        cpt_path = args['outf'] + "/train/" + date_time + "/cpt_pth/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(cpt_path):
            os.makedirs(cpt_path)
    
    # Init loggers
    if rank == 0:
        writer, logger = init_logger(args, log_path)


    # -----> Sec 1 Data <-----
    if rank == 0:
        logger.info("----------> Training HSV model <----------")
        logger.info(">> Settings: ")
        for k, v in zip(args.keys(), args.values()):
            logger.info('\t{}: {}'.format(k, v))
        logger.info('\n')
    
    # Train dataset
    if rank == 0:
        logger.info(">> Loading datasets ...")
    trainset_root = args['trainset_path']
    trainset_label = os.listdir(trainset_root)
    Trainset = []
    for i in range(len(trainset_label)):
        Trainset.append(HSVDataset(trainset_root, trainset_label[i]))
    Trainset = ConcatDataset(Trainset)
    Train_sampler = DistributedSampler(Trainset)

    # Validation dataset
    transform_val = v2.Compose([
        v2.CenterCrop((256, 256)),
        normalize(),
    ])
    valset_root = args['valset_path']
    valset_label = os.listdir(valset_root)
    Valset = []
    for i in range(len(valset_label)):
        Valset.append(HSVDataset_val(valset_root, valset_label[i], transform=transform_val))
    Valset = ConcatDataset(Valset)
    Val_sampler = DistributedSampler(Valset)
    
    # Dataloader
    Train_batch_sampler = BatchSampler(Train_sampler, args['batch_size'], drop_last=True)
    
    # num_workers = min([os.cpu_count(), args['batch_size'] if args['batch_size'] > 1 else 0, 8])
    num_workers = 12
    if rank == 0:
        logger.info("Using {} dataloader workers every process".format(num_workers))
        logger.info("Using {} world size".format(args['world_size']))
    
    Train_loader = DataLoader(dataset=Trainset, batch_sampler=Train_batch_sampler, 
                              pin_memory=True, num_workers=num_workers)
    Val_loader = DataLoader(dataset=Valset, batch_size=args['batch_size'], 
                            sampler=Val_sampler, pin_memory=True, num_workers=num_workers, drop_last=True)

    # Init masks
    masks = init_masks(args).to(device)
    
    
    # Two pre-computed matrices 
    H_mtx, A_mtx = get_H_A(args, device)
    
    
    # -----> Sec 2 Model <-----
    RESUME = args['resume']
    model = CSVSUF(args).to(device)

    if RESUME:
        path_checkpoint = args['pretrained_path']
        checkpoint = torch.load(path_checkpoint, map_location=device, weights_only=True)
        model_cpt = checkpoint['net']
        load_state_dict = OrderedDict()
        for k, v in model_cpt.items():
            load_key = k.replace("module.", "")
            load_state_dict[load_key] = v
        model.load_state_dict(load_state_dict, strict=True)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pth")
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    model = DDP(model, device_ids=[opt.gpu_id], find_unused_parameters=True)
    
    # -----> Sec 3 Metric <-----
    mse = torch.nn.MSELoss().to(device)
    
    
    # -----> Sec 4 Learning Rule <-----
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'] * math.sqrt(args['world_size']), betas=(0.9, 0.999))
    if args['scheduler'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])
    
    if RESUME:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_schedule'])
    
    # -----> Training <-----
    if rank == 0:
        logger.info(">> Start training: ")
        
    if RESUME:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 1
    
    psnr_max = 0
    for epoch in range(start_epoch, args['train_epochs'] + 1):
        
        # Train
        Train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        if is_main_process():
            Train_loader = tqdm(Train_loader, leave=False, colour='RED')
        
        start_time = time.time()
        
        optimizer.zero_grad()
        for i, data in enumerate(Train_loader):
            gt_batch = data
            gt = gt_batch.to(device).float() # gt: [B F C H W]
            
            # Create measurements
            input_meas = init_meas(gt, masks, device)
            model_out = model(input_meas, masks, H_mtx, A_mtx)
            
            # FLOPs and Params
            # if rank == 0:
            #     if i == 0 and epoch == 1:
            #         flops = FlopCountAnalysis(model, (input_meas, masks, H_mtx, A_mtx))
            #         params = sum(p.numel() for p in model.parameters())
            #         flops_G = flops.total() / 10**9
            #         params_mb = params * 4 / (1024 ** 2)
            #         logger.info(f"<-- MODEL --> FLOPs: {flops_G:.2f}G Params: {params_mb:.2f}MB")
            
            loss_train = torch.sqrt(mse(model_out, gt))
            loss_train.backward()
            
            epoch_loss += loss_train.data
            
            optimizer.step()
            optimizer.zero_grad()
            
        end_time = time.time()
        epoch_loss = reduce_value(epoch_loss, average=False)
        
        if rank == 0:
            logger.info("===> Epoch [{}/{}]. Avg. loss: {:.5f} Time: {:.2f}".
                    format(epoch, args['train_epochs'], epoch_loss / len(Trainset), (end_time - start_time)))
            writer.add_scalar('Training Avg. loss', epoch_loss / len(Trainset), epoch)
            
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        scheduler.step()
        
        # lr check
        if epoch % 50 == 0:
            if rank == 0:
                logger.info("> Learning rate: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        # Validation
        if epoch % args['val_interval'] == 0:
            
            if rank == 0:
                epoch_loss_val = 0
                psnr_list, ssim_list =  [], []
                total_time = 0
                
                model.eval()
                with torch.no_grad():
                    if is_main_process():
                        Val_loader = tqdm(Val_loader, leave=False, colour='RED')
                    for _, data in enumerate(Val_loader):
                        gt_batch = data
                        gt = gt_batch.to(device).float()
                        
                        # Create measurements
                        input_meas = init_meas(gt, masks, device)
                        
                        # Time analysis
                        start_time = time.time()
                        model_out = model(input_meas, masks, H_mtx, A_mtx)
                        end_time = time.time()
                        total_time += end_time - start_time
                        
                        # Loss
                        loss_val = torch.sqrt(mse(model_out, gt))
                        epoch_loss_val += loss_val.data
                        
                        # PSNR and SSIM
                        psnr_val = comp_psnr(model_out, gt)
                        psnr_list.append(psnr_val.detach().cpu().numpy())
                        ssim_val = comp_ssim(model_out, gt)
                        ssim_list.append(ssim_val)
                        
                psnr_mean = np.mean(np.asarray(psnr_list)) # per frame per channel
                ssim_mean = np.mean(ssim_list)
                
                
                if device != torch.device("cpu"):
                    torch.cuda.synchronize(device)
                
                # Save
                if psnr_mean > psnr_max:
                    psnr_max = psnr_mean
                    if psnr_max > 28:
                        # Model parameters
                        model_save_pth = model_path + "model_epoch_{}_psnr_{:.2f}.pth".format(epoch, psnr_max)
                        torch.save(model.state_dict(), model_save_pth)
                        logger.info("Model saved to {}".format(model_save_pth))
                            
                        # Checkpoint parameters
                        cpt_save_pth = cpt_path + "cpt_epoch_{}.pth".format(epoch)
                        checkpoint = {
                            "net": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "lr_schedule": scheduler.state_dict()
                        }
                        torch.save(checkpoint, cpt_save_pth)
                        logger.info("Checkpoint saved to {}".format(cpt_save_pth))
                        
                logger.info("\tValidation results: Avg. loss: {:.5f} PSNR: {:.2f} SSIM: {:.3f} Time: {:.2f}".
                        format(epoch_loss_val / len(Valset), psnr_mean, ssim_mean, total_time))
                writer.add_scalar('Validation Avg. loss', epoch_loss_val / len(Valset), epoch)
                writer.add_scalar('PSNR', psnr_mean, epoch)
                writer.add_scalar('SSIM', ssim_mean, epoch)
            
    if rank == 0:
        writer.close()
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    
    cleanup()


if __name__ == "__main__":
    main(**vars(opt))
    