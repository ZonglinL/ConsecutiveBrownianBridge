import os
import random
from flolpips.flolpips import Flolpips
import torch
from tqdm.autonotebook import tqdm
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_metric = Flolpips().to(device)


@torch.no_grad()
def calc_FLOLPIPS(data_dir, gt_dir,cond_dir, num_samples=1):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    total_lpips_distance = 0
    for i in tqdm(range(total), total=total, smoothing=0.01):
        gt_name = os.path.join(gt_dir, f'GT_{str(i)}.png')
        gt_img = 0.5 + lpips.im2tensor(lpips.load_image(gt_name)).to(torch.device('cuda:0'))/2 ## loaded as -1,1; flolpips need 0,1
        for j in range(num_samples):
            img_name = os.path.join(os.path.join(data_dir,f'{j}', f'sample_from_next{str(i)}.png'))
            img_calc = 0.5 + lpips.im2tensor(lpips.load_image(img_name)).to(torch.device('cuda:0'))/2
            
            prev_name = os.path.join(os.path.join(cond_dir, f'previous_frame{str(i)}.png'))
            prev_calc = 0.5 + lpips.im2tensor(lpips.load_image(prev_name)).to(torch.device('cuda:0'))/2
            
            next_name = os.path.join(os.path.join(cond_dir, f'next_frame{str(i)}.png'))
            next_calc = 0.5 + lpips.im2tensor(lpips.load_image(next_name)).to(torch.device('cuda:0'))/2

            current_lpips_distance = eval_metric.forward(prev_calc,next_calc, gt_img, img_calc)
            total_lpips_distance = total_lpips_distance + current_lpips_distance

    avg_lpips_distance = total_lpips_distance / (total * num_samples)
    print(f'flolpips_distance: {avg_lpips_distance}')
    return avg_lpips_distance 



