from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mri-denoiser",
    version="0.1.0",
    author="Arastu Thakur",
    author_email="arustuthakur@gmail.com",
    description="A package for denoising MRI images using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arastuthakur/mri-denoiser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.19.0",
        "tqdm>=4.60.0",
        "pillow>=8.0.0",
        "seaborn>=0.11.0"
    ],
    include_package_data=True,
    package_data={
        'mri_denoiser': ['models/*.pth'],
    },
) 