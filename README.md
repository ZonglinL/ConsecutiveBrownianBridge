# Frame Interpolation with Consecutive Brownian Bridge

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-xxxx.xxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxx)&nbsp;
</div>

<p align="center" style="font-size:1em;">
  <a href="https://zonglinl.github.io/videointerp/">Frame Interpolation with Consecutive Brownian Bridge</a>
</p>

<p align="center">
<img src="images/Teaser.jpg" width=95%>
<p>

## Overview
We design the Consecutive Brownian Bridge Diffusion that transits among three frames specifically for the frame interpolation task and takes advangtage of optical flow estimation in the autoencoder part. **The Consecutive Brownian Bridge Diffusion** reduce cumulative variance during sampling, which is prefered in VFI becuase there is a *deterministic groundtruth* rather than *a diverse set of images*. **The autoencoder with flow estimation** improves the visual quality of frames decoded from the latent space.

<p align="center">
<img src="images/overview.jpg" width=95%>
<p>

## Inference

Please install necessary packages in requirements.txt, then run:

```
python interpolate.py --resume_model path_to_model_weights --frame0 path_to_the_previous_frame --frame1 path_to_the_next_frame
```

The weights of of our trained model can be downloaded <a href="https://zonglinl.github.io/videointerp/">here</a>.

## Training and Evaluating

This part will be released after paper is accepted
