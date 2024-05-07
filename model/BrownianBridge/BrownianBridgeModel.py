import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler


class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()
        self.next_frame = False

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1]) ## left shifted mt

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var ## delta_t in the paper (variance of BB)
        variance_tminus = np.append(0., variance_t[:-1]) ## left shifted delta_t
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2 ## delta t|t-1
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t ## delta t in the reverse process
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y,z, context_y=None,context_z = None):
        ## context is default to None
        ## x is the target to sample (interpolated frame)

        
        if self.condition_key == "nocond":
            context_y = None
            context_z = None
        else:
            context_y = y if context_y is None else context_y
            context_z = z if context_z is None else context_z
            context_y = torch.cat([context_y,context_z],dim = 1).detach()
            context_z = context_y.clone().detach()
        
        
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.bi_p_losses(x, y, z, context_y,context_z, t) 

    def compute_loss(self,x,y,loss_weights = 1):
        diff = x - y
        if self.loss_type == 'l1':
            diff = diff.abs()
        else:
            diff = diff.pow(2.)
        diff = diff*loss_weights
        return diff.mean()


    def bi_p_losses(self, x, y, z, context_y,context_z, t, noise=None):
        """
        model loss
        :param x: encoded x current frame
        :param y: encoded y (previous frame)
        :param z: encoded z (next frame)
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x.shape
        noise = default(noise, lambda: torch.randn_like(x))
        loss_weights = 1
        var_t = extract(self.variance_t, t, x.shape)
        tmp = var_t
        snr = torch.sqrt(1/tmp)
        loss_weights = snr.clamp_(max = 5)
        
        if self.next_frame:
            ## if we do next frame prediction, we diffuse from z to x to y
            x_t_1, objective_1 = self.q_sample(x, y, t, noise)
            x_t_2,objective_2 = self.q_sample(z, x, t, noise)
        else:
            ## otherwise, we diffuse from x to y and x to z
            x_t_1,objective_1 = self.q_sample(x, y, t, noise) ## from x to y

            x_t_2,objective_2 = self.q_sample(x, z, t, noise) ## from x to z
        if self.next_frame:
            ## next frame prediction will only condition on previous frame instead of bidirectional
            objective_recon_1 = None
            objective_recon_2 = self.denoise_fn(x_t_2, x_t_1, timesteps=t, context=context_y)
        else:
            if np.random.rand()>0.5:
                objective_recon_2 = self.denoise_fn(x_t_2, x_t_1, timesteps=self.num_timesteps + (t+1), context=context_y)
            
                ## when we predicting noise from the next frame, change the context to next frame
            else:
                objective_recon_2 = self.denoise_fn(x_t_1, x_t_2, timesteps=self.num_timesteps - (t+1), context=context_z)
                objective_2 = objective_1
            
        if self.next_frame:
            recloss = self.compute_loss(objective_2,objective_recon_2,loss_weights)
        else:
            recloss = self.compute_loss(objective_2,objective_recon_2,loss_weights) #+ self.compute_loss(objective_1,objective_recon_1,loss_weights)
        
        '''
        if self.loss_type == 'l1':
            if self.next_frame:
                #recloss = (objective_2 - objective_recon_2).abs().mean()
                recloss = self.compute_loss(objective_2,objective_recon_2,loss_weights)
            else:
                recloss = (objective_1 - objective_recon_1).abs().mean() + (objective_2 - objective_recon_2).abs().mean()
        elif self.loss_type == 'l2':
            if self.next_frame:
                recloss = F.mse_loss(objective_2, objective_recon_2)
            else:
                recloss = F.mse_loss(objective_1, objective_recon_1) + F.mse_loss(objective_2, objective_recon_2)
        else:
            raise NotImplementedError()
        '''
        if self.next_frame:
            x0_recon = self.predict_x0_from_objective(x_t_2, z, t, objective_recon_2)
        else:
            #x0_recon = self.predict_x0_from_objective(x_t_1, y, t, objective_recon_1)
            x0_recon = self.predict_x0_from_objective(x_t_2, y, t, objective_recon_2)

            """
            x0_recon_next = self.predict_x0_from_objective(x_t_2, y, t, objective_recon_2)
            consistent_loss = self.compute_loss(x0_recon,x0_recon_next,loss_weights)
            """
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict


    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)
        x_t = (1. - m_t) * x0 + m_t * y + sigma_t * noise

        if self.objective == 'grad':
            #objective = m_t * (y - x0) + sigma_t * noise
            objective = x_t - x0
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        elif self.objective == 'BB':
            objective = x_t - x0
        else:
            raise NotImplementedError()

        return (
            x_t,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon

        elif self.objective == 'BB':
            x0_recon = -objective_recon + x_t ## if predicting xt - x0
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def bi_p_sample(self, x_t, cond, y, context, i, clip_denoised=False,is_z = False):
        ## xt the current denoised steps
        ## y the starting point of this path
        ## cond xt at another path
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            if self.next_frame == False:
                if is_z:
                    timestep = self.num_timesteps + (t + 1)
                else:
                    timestep = self.num_timesteps - (t + 1)
            objective_recon = self.denoise_fn(x_t,cond, timesteps=timestep, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
            if self.next_frame == False:
                if is_z:
                    timestep = self.num_timesteps + (t + 1)
                else:
                    timestep = self.num_timesteps - (t + 1)

            objective_recon = self.denoise_fn(x_t,cond, timesteps=timestep, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            noise = torch.randn_like(x_t)

            if self.objective == 'BB':
                step = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                step_prev = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
                
                #tn = extract(self.variance_t, step, x_t.shape) 
                #tnk = extract(self.variance_t, step_prev, x_t.shape)
                tn = torch.tensor(self.steps[i],device=x_t.device,dtype=torch.long)/self.num_timesteps
                tnk = torch.tensor(self.steps[i+1],device=x_t.device,dtype=torch.long)/self.num_timesteps
                one_ov_t = (tn - tnk)/tn
                sigma_t = torch.sqrt(tnk - (tnk.pow(2.)/tn)) ## approx. 0
                return x_t - one_ov_t*objective_recon + sigma_t*noise,x0_recon ## xt - (tn - tn-k)/tn * (x_t -x_0)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def bi_p_sample_loop(self, y,z, context_y=None,context_z = None, clip_denoised=True, sample_mid_step=False):
        ## y: previous frame
        ## z: next frame / current frame if in next frame pred
        if self.condition_key == "nocond":
            context_y = None
            context_z = None
        else:
            context_y = y if context_y is None else context_y
            context_z = z if context_z is None else context_z
            context_y = torch.cat([context_y,context_z],dim = 1).detach()
            context_z = context_y.clone().detach()

        
        imgs_prev,imgs_next, one_step_imgs_prev,one_step_imgs_next = [y],[z], [],[]
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):

            img_next, x0_recon_next = self.bi_p_sample(x_t=imgs_next[-1],cond = imgs_prev[-1], y=z, 
            context=context_y, i=i, clip_denoised=clip_denoised,is_z = True)
            ## sample one step from the next/current frames condition on previous (at same t)
            
            if self.next_frame:
                t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                img_prev = self.self.q_sample(y, z, t)
                ## if next frame prediction, we compute the forward diffusion of previous frame to the current frame
            else:
                
                img_prev, x0_recon_prev = self.bi_p_sample(x_t=imgs_prev[-1],cond = imgs_next[-1], y=y, 
                context=context_z, i=i, clip_denoised=clip_denoised,is_z = False)
                ## otherwise we sample from the previous frame
                ## at this time if we want to add qkv condition, it should be next frame
            imgs_next.append(img_next)
            imgs_prev.append(img_prev)
            one_step_imgs_prev.append(x0_recon_prev)
            one_step_imgs_next.append(x0_recon_next)
        if sample_mid_step:
            return imgs_next,imgs_prev, one_step_imgs_next,one_step_imgs_prev
        else:
            return imgs_next[-1],imgs_prev[-1]


    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img



    @torch.no_grad()
    def sample(self, y,z, context_y=None, context_z=None, clip_denoised=True, sample_mid_step=False):
        ## y: previous frame
        ## z: next frame in interpolation, bad frame in inpainting, current frame in next frame prediction
        ## context will be concatenated to x, also cross_attended
        return self.bi_p_sample_loop(y, z, context_y,context_z, clip_denoised, sample_mid_step)