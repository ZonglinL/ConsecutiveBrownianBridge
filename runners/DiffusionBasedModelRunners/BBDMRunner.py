import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary
from pytorch_ssim import ssim_matlab as ssim_
import numpy as np
import math

@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            ## calculate the mean of latents
            x,y,z = batch
            x = x.to(self.config.training.device[0])
            y = y.to(self.config.training.device[0])
            z = z.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            y_latent = self.net.encode(y, cond=True, normalize=False)
            z_latent = self.net.encode(z, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            cond_mean = (y_latent.mean(axis=[0, 2, 3], keepdim=True) + z_latent.mean(axis=[0, 2, 3], keepdim=True))/2
            total_cond_mean = cond_mean if total_cond_mean is None else cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            x,y,z = batch
            x = x.to(self.config.training.device[0])
            y = y.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            y_latent = self.net.encode(y, cond=True, normalize=False)
            z_latent = self.net.encode(z, cond=True, normalize=False)

            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((torch.cat([y,z],dim = 0) - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            # if batch_count >= max_batch_num:
            #     break
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        x,y,z = batch
        x = x.to(self.config.training.device[0])
        y = y.to(self.config.training.device[0])
        z = z.to(self.config.training.device[0])

        loss, additional_info = net(x, y, z)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        x, y, z= batch

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        y = y[0:batch_size].to(self.config.training.device[0])
        z = z[0:batch_size].to(self.config.training.device[0])

        grid_size = 4

        # samples, one_step_samples = net.sample(x_cond,
        #                                        clip_denoised=self.config.testing.clip_denoised,
        #                                        sample_mid_step=True)
        # self.save_images(samples, reverse_sample_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        #
        # self.save_images(one_step_samples, reverse_one_step_path, grid_size, save_interval=200,
        #                  writer_tag=f'{stage}_one_step_sample' if stage != 'test' else None)
        #
        # sample = samples[-1]
        sample_next, sample_prev = net.sample(y,z, clip_denoised=self.config.testing.clip_denoised)
        sample_next, sample_prev = sample_next.to('cpu'), sample_prev.to('cpu')
        ## sample mid step = False, only return sample from the next frame and past frame
        
        ## sample from next
        image_grid = get_image_grid(sample_next, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample_next.png'))
        
        ## sample from prev
        image_grid = get_image_grid(sample_prev, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample_prev.png'))


        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')

        ## save previous frame
        image_grid = get_image_grid(y.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'prev_frame.png'))


        ## save next frame
        image_grid = get_image_grid(z.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'next_frame.png'))


        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        #sample_num = len(test_loader)
        condition_path = make_dir(os.path.join(sample_path,f'condition'))
        gt_path = make_dir(os.path.join(sample_path,f'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        for j in range(sample_num):
            k = 0
            result_path_j = make_dir(os.path.join(result_path, f'{j}'))
            PSNR = 0
            ssim = 0
            mse = 0

            for test_batch in pbar:
                x,y,z = test_batch
                batch_size = x.shape[0]
                
                x = x.to(self.config.training.device[0])
                y = y.to(self.config.training.device[0])
                z = z.to(self.config.training.device[0]) 
                

                '''
                Encode and Deocde, without sampling

                x_lat = self.net.encode(x,cond = False)
                y_lat,prev_phi = self.net.encode(y,cond = True)
                z_lat,next_phi = self.net.encode(z,cond = True)
                sample_next = self.net.decode(x_lat,y,z,prev_phi,next_phi)
                sample_prev = sample_next
                '''
                
                sample_next,sample_prev = net.sample(y,z, clip_denoised=False)
                
                for i in range(batch_size):
                    condition_prev = y[i].detach().clone()
                    condition_next = z[i].detach().clone()
                    gt = x[i]
                    result_next = sample_next[i].detach()
                    result_prev = sample_prev[i].detach()

                    save_single_image(condition_prev, condition_path, f'previous_frame{k}.png', to_normal=to_normal)
                    save_single_image(condition_next, condition_path, f'next_frame{k}.png', to_normal=to_normal)
                    save_single_image(gt, gt_path, f'GT_{k}.png', to_normal=to_normal) 

                    save_single_image(result_prev, result_path_j, f'sample_from_prev{k}.png', to_normal=to_normal)
                    save_single_image(result_next, result_path_j, f'sample_from_next{k}.png', to_normal=to_normal)
                    k += 1
                    
                    pred = ((result_next/2) + 0.5).unsqueeze(0)
                    gt = ((gt/2) + 0.5).unsqueeze(0)
                    ssim += ssim_(gt.cpu(), pred.cpu().clamp_(min = 0,max = 1),val_range = 1)
                    
                    pred = pred.cpu().numpy()
                    gt = gt.cpu().numpy()
                    mse = ((pred - gt)**2).mean()
                    PSNR += -10*math.log10(mse)
                    #ssim += SSIM_func.forward(gt,pred)
            
        print(f"PSNR: {PSNR/k}")
        print(f"SSIM: {ssim/k}")