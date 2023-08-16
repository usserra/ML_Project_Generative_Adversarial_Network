# ML_Project_Generative_Adversarial_Network
*ML Assignment for Gizatech*
## Introduction
This project aims to create a successful machine learning model to generate new images from given images in the style of Claude Monet. 

GANs consist of two main neural network parts - generative model and the discriminator model- Generative models job is to generate new images that are ideally indisguisable from the ones in the training dataset. Generative model architecture is originally an unsupervised learning example where the model tries to discover the patterns itself wihthout the coder giving it the "correct" answer. But when combined with a discriminator model the situation turns into a supervised learning area. The job of the discriminator is to take generated images from the generator model and try to guess whether it is real or fake. 
During training if generated images are not well discriminated by the discriminator is penalized and its weights are changed whereas if discriminator is doing a good job at recognizing what is fake or real generator model is penalized and its weights are changed. Optimal situation is where discriminator outputs unsure i.e. 50% for real or fake.  

A CycleGAN model will be used for this image-to-image translation problem where the photos will be tranformed into Monet images and vice-versa which ensures better training

## Projects: 
GAN -> 
GAN2 -> 