import os
import re
import scipy.io as scio
import numpy as np

Model = "CSVSUF-9stg"
Scene = "automobile6"
path = "experiments/CSVSUF-9stg/test/automobile6/2025_04_09_15_54_03/results"
pattern = r"Test_fr(\d+)-\d+_(\d+\.\d+)_([\d]+\.[\d]+)\.mat"

entries = []
sam_list = []

filenames = [f for f in os.listdir(path) if f.endswith('.mat')]

for filename in filenames:
    match = re.match(pattern, filename)
    if match:
        start_frame = int(match.group(1))
        psnr = float(match.group(2))
        ssim = float(match.group(3))
        entries.append((start_frame, psnr, ssim))

        # SAM
        data = scio.loadmat(os.path.join(path, filename))
        gt = data['gt'].flatten()
        recon = data['recon'].flatten()

        dot = np.dot(gt, recon)
        norm_gt = np.dot(gt, gt)
        norm_recon = np.dot(recon, recon)
        sam_angle = np.arccos(dot / np.sqrt(norm_gt * norm_recon))
        sam_list.append(sam_angle)

# Sort
entries.sort(key=lambda x: x[0])
psnr_list = [e[1] for e in entries]
ssim_list = [e[2] for e in entries]

# Save
output_file = os.path.join(path, f"metrics_list_{Model}_{Scene}.mat")
scio.savemat(output_file, {
    'PSNR_list': np.array(psnr_list),
    'SSIM_list': np.array(ssim_list),
    'SAM_list': np.array(sam_list)
})

