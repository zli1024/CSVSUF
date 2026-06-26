import os
import time
import datetime
from tqdm import tqdm
from torchvision.transforms import v2
from torch.utils.data import BatchSampler, DataLoader, ConcatDataset
from collections import OrderedDict

from utils import *
from options import opt
from architectures.CSVSUF import CSVSUF

def main(**args):
    r""" Main testing
    """
    # -----> Sec 0 Preprocessing <-----
    # Set seed
    set_seed(args['seed'])
    
    # Define devices
    if torch.cuda.is_available() is False:
        raise EnvironmentError("No GPU available!")
    device = torch.device(opt.device)
    
    # Saving path
    date_time = str(datetime.datetime.now())
    date_time = time2filename(date_time)
    result_path = args['outf'] + date_time + "/results/"
    log_path = args['outf'] + date_time + "/logs/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    # Init loggers
    _, logger = init_logger(args, log_path)
    
    
    # -----> Sec 1 Data <-----
    logger.info("----------> Testing HSV model <----------")
    logger.info(">> Settings: ")
    for k, v in zip(args.keys(), args.values()):
        logger.info('\t{}: {}'.format(k, v))
    logger.info('\n')
    
    # Dataset
    logger.info(">> Loading datasets ...")
    
    transform_test = v2.Compose([
        v2.CenterCrop((256, 256)),
        normalize(),
    ])
    testset_root = args['testset_path']
    testset_label = os.listdir(testset_root)
    Testset = []
    for i in range(len(testset_label)):
        Testset.append(HSVDataset_val(testset_root, testset_label[i], transform=transform_test))
    Testset = ConcatDataset(Testset)
    
    # Dataloader
    num_workers = 12
    batch_sampler = BatchSampler(NonOverlappingSampler(Testset, num_frames_per_sample=args['temp_patch_size']),
                                 batch_size=args['batch_size'], drop_last=True)
    Test_loader = DataLoader(dataset=Testset, batch_sampler=batch_sampler,
                             pin_memory=True, num_workers=num_workers)
    
    # Init masks
    masks = init_masks(args).to(device)
    
    # Two pre-computed matrices 
    H_mtx, A_mtx = get_H_A(args, device)
    
    
    # -----> Sec 2 Model <-----
    model = CSVSUF(args).to(device)
    model_pth = args['model_path']
    model_cpt = torch.load(model_pth, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_cpt.items():
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    
    
    # -----> Sec 3 Metric <-----
    mse = torch.nn.MSELoss().to(device)
    
    
    # -----> Testing <-----
    logger.info(">> Start testing: ")
    
    total_loss = 0
    psnr_list, ssim_list = [], []
    total_time = 0
    
    model.eval()
    with torch.no_grad():
        Test_loader = tqdm(Test_loader, leave=False, colour='RED')
        for i, data in enumerate(Test_loader):
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
            loss = torch.sqrt(mse(model_out, gt))
            total_loss += loss.data
            
            # PSNR and SSIM
            psnr = comp_psnr(model_out, gt)
            psnr_list.append(psnr.detach().cpu().numpy())
            ssim = comp_ssim(model_out, gt)
            ssim_list.append(ssim)
            
            # Info
            logger.info("\t Frame{}-{} results: Avg. loss: {:.5f} PSNR: {:.2f} SSIM: {:.3f}".
                        format(i*(args['batch_size']*args['temp_patch_size']) + 1,
                               (i+1)*args['batch_size']*args['temp_patch_size'],
                               loss, psnr_list[i], ssim_list[i]))
            
            # Save results
            if args['if_save']:
                save_pth = result_path + "Test_fr{}-{}_{:.2f}_{:.3f}".format(
                    i*(args['batch_size']*args['temp_patch_size']) + 1,
                    (i+1)*args['batch_size']*args['temp_patch_size'],
                    psnr_list[i], ssim_list[i]) + ".mat"
                scio.savemat(save_pth, {'recon': model_out.detach().cpu().numpy(),
                                        'gt': gt.detach().cpu().numpy()})
            
    psnr_mean = np.mean(np.asarray(psnr_list)) # per frame per channel
    ssim_mean = np.mean(ssim_list)
    
    logger.info("\tTesting results: total loss: {:.5f} Avg.PSNR: {:.2f} Avg.SSIM: {:.3f} Total Time: {:.2f}".
                format(total_loss, psnr_mean, ssim_mean, total_time))


if __name__ == "__main__":
    main(**vars(opt))