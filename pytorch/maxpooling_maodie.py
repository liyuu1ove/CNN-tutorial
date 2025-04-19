import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    img = Image.open(path).convert('L') 
    transform = transforms.Compose([
        transforms.ToTensor()           
    ])
    return transform(img).unsqueeze(0)  

def maxpooling():
    pooling = nn.MaxPool2d(kernel_size=2, stride=2)
    return pooling

def plot_images(original, result):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze(), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.title('Pooling Result')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":

    image_path = 'asset/maodie.jpg' 
    
    input_tensor = load_image(image_path)
    
    pooling_layer = maxpooling()
    
    with torch.no_grad():
        output_tensor = pooling_layer(input_tensor)
    
    output = output_tensor.squeeze().numpy()  
    output = (output - output.min()) / (output.max() - output.min()) 
    
    plot_images(input_tensor.squeeze().numpy(), output)