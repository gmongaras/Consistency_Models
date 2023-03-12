# Consistency models

I came across a paper called ["Consistency Models"](https://arxiv.org/abs/2303.01469) which came out recently. 

Diffusion models are currently state of the art (SOTA) for image generation, but a major issue with them is the generation time. Since diffusion models are trained to generate images over multiple steps (usually around 1000), it takes the same number of steps to generate a single image. There exists methods like DDIMs which speed up the process by skipping steps, but they can only go so far without degrading sample quality. So, sampling time is an issue.

![image](https://user-images.githubusercontent.com/43501738/224571851-b1662f64-a868-43f0-9a0d-a2080a65c7df.png)

Consistency models are a similar type of model to diffusion models and are proposed to replace diffusion models since consistency models appear to do as well as diffusion models, but can generate images in a single step.

Consistency models are trained to generate the original image, x_0 from any step in the "diffusion" process. Instead of having to go through all steps to get the image, we can just take a single step from noise to the output image. We can also improve sample quality by adding noise back and having the model predict the original image, x_0, but that isn't required to generate images.



# Project Description

Unfortunately, the code for Consistency Models is not currently released and I'm going to try to implement these types of models here.

Currently, I have implemented the following:
- Training a consistency model from scratch (Training Consistency Models in Isolation)
- Checkpointing

I hope to implement the following:
- Training using a diffusion model as guidance (Training Consistency Models via Distillation)
- Zero-shot image generation
