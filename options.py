""" Python file for configuring command-line options
"""

import argparse
from model_options import merge_csvsuf_opt

parser = argparse.ArgumentParser(description="HyperSpectral Video Reconstruction")

# Description
parser.add_argument("--desc", type=str, default="Testing (Rainystreet5): CSVSUF-3stg", help="Description for the experiments")

# Data settings
parser.add_argument("--data_root", type=str, default="./datasets/", help="datasets directory")
parser.add_argument("--outf", type=str, default="./experiments/", help="Results saving path")
parser.add_argument("--seed", type=int, default=369, help="Random seed")

# Device settings
parser.add_argument("--gpu_ids", type=str, default='0, 1')
parser.add_argument("--device", default="cuda", help="device id (i.e. 0 or 0,1 or cpu)")
parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

# Model settings
parser.add_argument("--model", type=str, default="CSVSUF", help="Specify which model to use")
parser.add_argument("--model_path", type=str, default="./model_zoo/CSVSUF-3stg/model_epoch_194_psnr_30.93.pth", help="The path of pretrained model")

# Training and validation
parser.add_argument("--batch_size", type=int, default=2, help="the batch size of sequences of frames[F C H W]")
parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
parser.add_argument("--train_epochs", "--ep", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lambda", "--lam", type=float, default=0.5, help="tunning parameter for data fidelity term")
parser.add_argument("--rho", type=float, default=0.5, help="tunning parameter for penalty term")
parser.add_argument("--num_iter", "--it", type=int, default=3, help="number of iterations for HQS algorithm")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--scheduler", type=str, default="MultiStepLR")
parser.add_argument("--milestones", type=int, default=[50, 100, 140, 170, 190], help="milestones for MultiStepLR")
parser.add_argument("--gamma", type=float, default=0.5, help="learning rate decay for MultiStepLR")
parser.add_argument("--val_interval", type=int, default=1, help="the interval time for validation")
parser.add_argument("--resume", action="store_true", help="if resume for training")

# Testing
parser.add_argument("--if_save", type=str, default="True", help="whether save the testing results (True/False)")

opt = parser.parse_known_args()[0]

if opt.model == "CSVSUF":
    parser = merge_csvsuf_opt(parser)
opt = parser.parse_known_args()[0]

# Others
opt.trainset_path = f"{opt.data_root}/HSV-train/"
opt.valset_path = f"{opt.data_root}/HSV-val/"
opt.testset_path = f"{opt.data_root}/HSV-test1/"
opt.masks_path = f"{opt.data_root}/Masks/"
opt.ckp_path = "experiments/CSVSUF-9stg/train/2025_03_17_10_07_44/cpt_pth/cpt_epoch_10.pth"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False