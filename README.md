# MRI Image Denoiser

A deep learning-based solution for denoising MRI images using an enhanced denoising autoencoder architecture with self-attention and residual blocks.

## Features

- Enhanced Denoising Autoencoder with self-attention mechanism
- Residual blocks for better feature preservation
- Skip connections for detailed reconstruction
- Lightweight model variant available for resource-constrained environments
- Support for both training and inference on MRI images
- Flexible dataset creation from various image formats

## Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
Pillow>=8.0.0
tqdm>=4.50.0
```

## Model Architecture

The project includes two model variants:

1. **Enhanced Denoising Autoencoder**
   - 3-level encoder-decoder architecture
   - Self-attention modules
   - Residual blocks
   - Skip connections
   - Dropout for regularization

2. **Light Denoising Autoencoder**
   - 2-level encoder-decoder architecture
   - Lightweight residual blocks
   - Skip connections
   - Suitable for faster inference

## Input Requirements

- Input images should be grayscale (single channel)
- Images will be automatically resized to match model requirements
- Supported formats: JPG, PNG
- Recommended minimum resolution: 128x128 pixels

## Usage

### Dataset Preparation

Use the provided dataset creator utility to prepare your training data:

```python
python dataset_creator.py --input_dir /path/to/images --output_dir /path/to/output --size 256
```

### Training

```python
python train.py --data_dir /path/to/processed/data --model enhanced --epochs 100
```

### Inference

```python
python inference.py --input_image path/to/image.jpg --model_path path/to/model.pth
```

## Model Parameters

- Input channels: 1 (grayscale)
- Initial features: 32 (Enhanced) / 16 (Light)
- Dropout rate: 0.1
- Activation: LeakyReLU (Enhanced) / ReLU (Light)
- Final activation: Sigmoid

## License

MIT License

## Citation

If you use this code for your research, please cite:

```
@software{mri_image_denoiser,
  title = {MRI Image Denoiser},
  author = {arastu},
  year = {2025},
  url = {https://github.com/arastuthakur/mri-image-denoiser}
}
``` 