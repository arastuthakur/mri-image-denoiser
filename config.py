class Config:
    # Training settings
    batch_size = 8  # Small batch size for CPU
    num_epochs = 50
    learning_rate = 0.001
    noise_level = 0.1
    
    # Model settings
    input_channels = 1
    image_size = 128
    
    # Dataset settings
    train_dir = 'data/train'
    val_dir = 'data/val'
    
    # Checkpoint settings
    checkpoint_dir = 'checkpoints'
    save_frequency = 5  # Save checkpoint every 5 epochs
    
    # Wandb settings
    project_name = 'mri-denoising'
    run_name = 'cpu-training'

    # Data Configuration
    DATA_PATH = "./data/samples"  # We'll store our small dataset here
    TRAIN_BATCH_SIZE = 4  # Reduced batch size for CPU
    VAL_BATCH_SIZE = 4
    NUM_WORKERS = 0  # For CPU training
    IMAGE_SIZE = 128  # Reduced image size for faster processing

    # Model Configuration
    # U-Net Autoencoder - Smaller architecture
    UNET_CHANNELS = [32, 64, 128]  # Reduced number of channels
    UNET_ATTENTION_LAYERS = [False, True, True]
    
    # DDPM Configuration
    TIMESTEPS = 100  # Reduced timesteps for faster training
    BETA_START = 0.0001
    BETA_END = 0.02
    
    # Training Configuration
    WEIGHT_DECAY = 1e-4
    DEVICE = "cpu"  # Set to CPU
    
    # Noise Configuration
    GAUSSIAN_NOISE_MEAN = 0.0
    GAUSSIAN_NOISE_STD = 0.1
    MOTION_BLUR_KERNEL_SIZE = 7  # Reduced kernel size
    
    # Logging Configuration
    WANDB_PROJECT = "mri-denoising-cpu"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_INTERVAL = 10 