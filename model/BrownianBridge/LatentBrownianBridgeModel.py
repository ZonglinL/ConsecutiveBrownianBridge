import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQFlowNetInterface


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQFlowNetInterface(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan ## VQGAN quantization
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams)) ## interpolation
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, y, z,  context_y=None,context_z = None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            y_latent,_ = self.encode(y, cond=True)
            z_latent,_ = self.encode(z, cond = True)
        
        context_y = self.get_cond_stage_context(y)
        context_z = self.get_cond_stage_context(z)
        return super().forward(x_latent.detach(), y_latent.detach(),z_latent.detach(),context_y,context_z)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            if self.condition_key == 'first_stage':
                context = self.encode(x_cond,cond = True)[0].detach()
            else:
                context = self.cond_stage_model(x_cond)
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        model = self.vqgan
        if cond:
            x_latent,ret = model.encode(x,ret_feature=cond)
            return x_latent,ret
        else:
            x_latent = model.encode(x,ret_feature=cond)
            return x_latent

    @torch.no_grad()
    def decode(self, x_latent, prev_img,next_img,prev_phi,next_phi):
        model = self.vqgan
        out = model.decode(x_latent,prev_img,next_img,prev_phi,next_phi)
        return out

    @torch.no_grad()
    def latent_p_sample_loop(self, y, z, y_ori,z_ori,context_y,context_z,clip_denoised=True, sample_mid_step=False):
        ## y: previous frame
        ## z: next frame / current frame if in next frame pred
        ## context y: end point y, spatial resclaer, first stage or none, same as z

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
        return imgs_next,imgs_prev, one_step_imgs_next,one_step_imgs_prev



    @torch.no_grad()
    def sample(self, y, z, clip_denoised=False, sample_mid_step=False):
        y_latent,prev_phi = self.encode(y, cond=True)
        z_latent,next_phi = self.encode(z,cond = True)

        context_y = self.get_cond_stage_context(y)
        context_z = self.get_cond_stage_context(z)
        context_y = torch.cat([context_y,context_z],dim = 1).detach()
        context_z = context_y.clone().detach()

        """
        call the latent sample function
        """

        temp_next, temp_prev, one_step_temp_next,one_step_temp_prev = self.latent_p_sample_loop(y = y_latent,z =  z_latent,
                                            y_ori = y, z_ori = z,
                                            context_y = context_y,
                                            context_z = context_z,
                                            clip_denoised=clip_denoised,
                                            sample_mid_step=sample_mid_step)

        """
        decoding one step sampled latents to images at each t, both from previous and next frame

        """
        if sample_mid_step:
            one_step_samples_next,one_step_samples_prev,out_samples_next,out_samples_prev = [],[], [],[]
            for i in tqdm(range(len(one_step_temp_next)), initial=0, desc="save one step sample mid steps",
                            dynamic_ncols=True,
                            smoothing=0.01):
                with torch.no_grad():
                    out_next = self.decode(one_step_temp_next[i].detach(),y,z,prev_phi,next_phi)
                    out_prev = self.decode(one_step_temp_prev[i].detach(), y,z,prev_phi,next_phi)

                one_step_samples_next.append(out_next.to('cpu'))
                one_step_samples_prev.append(out_prev.to('cpu'))

                with torch.no_grad():
                    out_next = self.decode(temp_next[i].detach(), y,z,prev_phi,next_phi)
                    out_prev = self.decode(temp_prev[i].detach(), y,z,prev_phi,next_phi)
                out_samples_next.append(out_next.to('cpu'))
                out_samples_prev.append(out_prev.to('cpu'))

        
            return out_samples_next,out_samples_prev, one_step_samples_next,one_step_samples_prev
        else:
            """
            decoding sampled latents to images at each t, both from previous and next frame

            """
            with torch.no_grad():
                out_next = self.decode(temp_next[-1].detach(), y,z,prev_phi,next_phi)
                out_prev = self.decode(temp_prev[-1].detach(), y,z,prev_phi,next_phi)

            return out_next,out_prev

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
