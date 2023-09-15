# ML_Project_Generative_Adversarial_Network
*ML Assignment for Gizatech*
## Introduction
This project aims to create a successful machine learning model to generate new images from given images in the style of Claude Monet. 

GANs consist of two main neural network parts - generative model and the discriminator model- Generative models job is to generate new images that are ideally indisguisable from the ones in the training dataset. Generative model architecture is originally an unsupervised learning example where the model tries to discover the patterns itself wihthout the coder giving it the "correct" answer. But when combined with a discriminator model the situation turns into a supervised learning area. The job of the discriminator is to take generated images from the generator model and try to guess whether it is real or fake. 
During training if generated images are not well discriminated by the discriminator is penalized and its weights are changed whereas if discriminator is doing a good job at recognizing what is fake or real generator model is penalized and its weights are changed. Optimal situation is where discriminator outputs unsure i.e. 50% for real or fake.  

A CycleGAN model will be used for this image-to-image translation problem where the photos will be tranformed into Monet images and vice-versa which ensures better training.

## Contents: 
* This project consists of a local computer file, **GAN_Image_Preprocessing**, and a **Kaggle** folder containing the **Kaggle_Project** file which is the file that was worked on in a Kaggle server. 
* Certain dataset folders are not shown in GitHub. 
* .gitignore to hide folders/files from being visible in GitHub.
* A poetry.lock and a pyproject.toml file is created thorugh Poetry + Pyenv and are used to manage dependencies and versions in the project.

## Setup and Installation
To run the project, follow these steps:

For local computer:
1. Install Pyenv and Poetry.
2. Clone the repository and navigate to the GAN_Project folder.
3. Create a virtual environment: pyenv virtualenv 3.9.0 ml_project_env.
4. Activate the environment: pyenv activate ml_project_env.
5. Install project dependencies: poetry install.

For Kaggle server:
1. Setup a Kaggle account
2. Go to "project" notebook (https://www.kaggle.com/code/serraus/project/notebook)
3. Click "edit" 

## Libraries
python = ">=3.9, <3.12"
numpy = ">=1.19,<=1.24.3"
pandas = "^2.0.3"
keras = "2.2.4"
tensorflow = ">=1.13.1"
matplotlib = "^3.7.2"
ipykernel = "^6.25.1"
tensorflow-addons = "^0.21.0"
pillow = "^10.0.0"

## Dataset Description
The initial dataset was obtained from Kaggle. It had a 4 folders, containing 300 images of Monet paintings, and 7038 images. The other two folders had special tensorflow type image data which was not used. 
The image sizes were (256, 256, 3) RGB images. 

## Pipeline
1. Creating a virtual environment thorugh Pyenv and Poetry
2. Downloading Data from Kaggle
3. Loading the data
4. Visualising data using PIL library
5. Cropping Monet painting images as a form of data preprocessing
6. Normalising pixel values as a form of data preprocessing
7. Turn images into numpy arrays
8. Download them in a pickle file (This step is only necessray if one wants to transfer data into a different environment)
9. If in the new environment, load the pickle files 
10. Build the model architecture
11. Apply training
12. Save and compare results

## Problems 
Throughout the project there were mainly technical problems. The local computer had hardware memory issues as well as smalll CPU/RAM therefore after a couple of tries it was decided to use a public server to continue the project. Both kaggle and Google Collab was tested and Kaggle was finally decided on due to its high RAM capacity. The whole project had to get transferred to Kaggle and again problems with low memory came up. Then it was decided to do the data preprocessing on local computer whihc was slower but at least could be done. Later on ready data was transferred to Kaggle for training. During training TPU accelerator function in Kaggle was discovered and used due to its RAM being 330 GB. However due to the small batch size TPU was not allocated, however the project could still benefit from its memory while using the CPU to conduct training. There were still some problems with time limit for TPU usage (9 hours daily, 20 hours weekly).   
## CycleGAN Network 
The cycleGAN Network used in this project was taken from the original cycleGAN paper by .. 
The CycleGAN Network is explained in this notion page:
https://www.notion.so/gizadocs/ML_Project_Generative_Adversarial_Network-d62933971d3d44b692275906be57012b

- The discriminator models consist of Convolutional2D layers, with 64-128-256-512-512 and 1(Patch) filters. As an optimizer Adam was picked which is a common choice. Leaky Relu as activation function was picked in order to overcome linearity problem when dealing with complex functions and also becasue it helps big models to converge faster. Group normalization was applied due to small batch sizes and to ensure training stability since it does not realy on batch size instead of its own groups when calculating certain statistics. 

- The generator models consist of a Convolutional2D encoder part with 64-128-256, followed by a Convolutional2DTranspose decoder part with 128-64-3 filters. Relu activation function was used. Group normalization was applied. 

- Composite model comobines the outputs of both the generators and the discriminators in order to calculate the losses (mse, mae) and update the weights during training. As an optimizer Adam was picked which is a common choice. 

* While in training the composite models for both turning Monet into photo and photo into Monet is trained as well as the both discriminators. 

The models used are pretty deep in order for them to capture the complexity in images. This type of architecture is mainly preferred in computer vision tasks. To learn more about the CycleGANs please read the notion page.   

## Outputs
![Original Image](https://user-images.githubusercontent.com/123895232/268352620-82f0bff5-f20d-46a1-b78b-40e4df5e2c3b.png)

![Image1](https://user-images.githubusercontent.com/123895232/268350075-3ad82714-903a-411a-ad3d-5f9b48a2710a.png)

This image was formed after training for 46 steps. 

![Image2](https://user-images.githubusercontent.com/123895232/268350449-ceb804ff-e593-43ea-88cc-3a9e27111e6a.png)

This image was formed after training for 575 steps.

![Image3](https://user-images.githubusercontent.com/123895232/268350708-95c8030b-8841-4747-a050-49429836749f.png)

This image was formed after training for 2300 steps.

After analyising the images, one can say that the generator is changing the images while preserving its own characteristics. By looking at the images one can still say that it is the image of a cliff. This together with low cycle loss values proves that the generator is working fine in that sense. However one can notice that the generated images are not in the style of target domain - Monet-. This together with relatively high values for adversarial loss shows that the model still couldn't "learn" the details of Monet images. However it is visible that the model tried to make the lines of the objects in the photo a little smoother and made the colors less vibrant which is the case for Monet paintings. One can say that maybe after a lot more training the images could be in the form of the target domain. Or maybe one should change the architecture of the model and the paramters to see if it helps. 
*Due to time and computational constraints it was decided to leave the model as it is and at this level of training*

## Contributors 
@EgeAtesalp (Reviewer)