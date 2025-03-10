import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from model import EnhancedDenoisingAutoencoder, LightDenoisingAutoencoder
import numpy as np

def load_model(model_path, model_type='enhanced'):
    """Load the trained model."""
    if model_type.lower() == 'enhanced':
        model = EnhancedDenoisingAutoencoder()
    else:
        model = LightDenoisingAutoencoder()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image_path, size=256):
    """Load and preprocess the input image."""
    # Open and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    # Transform image
    img_tensor = transform(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img

def save_result(original_img, denoised_tensor, output_path):
    """Save the original and denoised images side by side."""
    # Convert denoised tensor to numpy array
    denoised_img = denoised_tensor.squeeze().cpu().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot denoised image
    ax2.imshow(denoised_img, cmap='gray')
    ax2.set_title('Denoised Image')
    ax2.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='MRI Image Denoiser Inference')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model_type', type=str, default='enhanced',
                        choices=['enhanced', 'light'],
                        help='Type of model to use (enhanced or light)')
    parser.add_argument('--size', type=int, default=256,
                        help='Size to resize input image (default: 256)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type)
    model = model.to(args.device)
    
    # Process input image
    print(f"Processing input image {args.input_image}...")
    img_tensor, original_img = process_image(args.input_image, args.size)
    img_tensor = img_tensor.to(args.device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        denoised = model(img_tensor)
    
    # Save results
    output_path = output_dir / f"denoised_{Path(args.input_image).stem}.png"
    print(f"Saving results to {output_path}...")
    save_result(original_img, denoised, output_path)
    
    print("Done!")

if __name__ == '__main__':
    main() 