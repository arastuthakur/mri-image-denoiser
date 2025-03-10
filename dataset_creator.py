import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))

def process_image(image_path, size):
    """Process a single image file."""
    try:
        # Open image and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Create transform pipeline
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
        # Transform image
        img_tensor = transform(img)
        
        return img_tensor, True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, False

def find_all_images(directory):
    """Recursively find all image files in directory and subdirectories."""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    return image_files

def create_dataset(args):
    """Create processed dataset from raw images."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    print("Finding image files...")
    image_files = find_all_images(args.input_dir)
    
    if not image_files:
        print("No image files found in the specified directory!")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Process images
    processed_count = 0
    failed_files = []
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Process image
        img_tensor, success = process_image(image_path, args.size)
        
        if success:
            # Create output filename
            output_filename = f"image_{processed_count:05d}.pt"
            output_path = output_dir / output_filename
            
            # Save processed tensor
            torch.save(img_tensor, output_path)
            processed_count += 1
        else:
            failed_files.append(image_path)
    
    # Save dataset info
    info = {
        'total_images': len(image_files),
        'processed_images': processed_count,
        'failed_images': len(failed_files),
        'image_size': args.size,
        'failed_files': failed_files
    }
    
    # Save dataset info
    with open(output_dir / 'dataset_info.txt', 'w') as f:
        f.write("Dataset Information:\n")
        f.write(f"Total images found: {info['total_images']}\n")
        f.write(f"Successfully processed: {info['processed_images']}\n")
        f.write(f"Failed to process: {info['failed_images']}\n")
        f.write(f"Image size: {info['image_size']}x{info['image_size']}\n")
        if failed_files:
            f.write("\nFailed files:\n")
            for file in failed_files:
                f.write(f"- {file}\n")
    
    print("\nDataset creation completed!")
    print(f"Successfully processed {processed_count} images")
    print(f"Failed to process {len(failed_files)} images")
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create MRI image dataset from raw images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing raw images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed dataset')
    parser.add_argument('--size', type=int, default=256,
                        help='Size to resize images to (default: 256)')
    
    args = parser.parse_args()
    create_dataset(args)

if __name__ == '__main__':
    main() 