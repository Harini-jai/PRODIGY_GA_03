import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMAGE_SIZE = 512 if torch.cuda.is_available() else 256

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content_img = load_image("images/content.jpg")
style_img = load_image("images/style.jpg")


vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

content_layer = '21'
style_layers = ['0', '5', '10', '19', '28']

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)   

def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in style_layers:
            features[name] = x
        if name == content_layer:
            features['content'] = x
    return features

target = content_img.clone().requires_grad_(True).to(device)

optimizer = optim.Adam([target], lr=0.003)

style_weight = 1e6
content_weight = 1

num_steps = 300
print("Starting Style Transfer...")

for step in range(num_steps):
    target_features = get_features(target, vgg)
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)

    # Content loss
    content_loss = torch.mean(
        (target_features['content'] - content_features['content']) ** 2
    )

    # Style loss
    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.mean((target_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}/{num_steps} | Total Loss: {total_loss.item():.2f}")

os.makedirs("output", exist_ok=True)

output_img = target.clone().detach().cpu().squeeze(0)
output_img = output_img.permute(1, 2, 0)

# De-normalize
output_img = output_img * torch.tensor([0.229, 0.224, 0.225]) + \
             torch.tensor([0.485, 0.456, 0.406])
output_img = output_img.clamp(0, 1)

plt.imsave("output/stylized_output.png", output_img.numpy())
print("Stylized image saved as output/stylized_output.png ")
