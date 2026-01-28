# PRODIGY_GA_03
# Neural Style Transfer using PyTorch

This project demonstrates Neural Style Transfer (NST) by applying the artistic style of one image to the content of another using deep learning techniques.

The implementation uses a pretrained VGG19 convolutional neural network to extract content and style features and blend them through optimization.

# Features

Applies artistic style to a content image

Uses pretrained VGG19 network

Computes content loss and style loss using Gram matrices

Generates a stylized output image

Runs on both CPU and GPU

Technologies Used

Python

PyTorch

Torchvision

Pillow (PIL)

Matplotlib


# How It Works

Loads content and style images

Extracts features using VGG19

Preserves content structure using content loss

Captures artistic patterns using style loss

Optimizes the target image iteratively

Saves the final stylized image

# How to Run

Install dependencies
pip install torch torchvision pillow matplotlib

Place images
Add content.jpg and style.jpg inside the images folder

Run the program
python nst.py

# Output
The stylized image is saved in the output folder
