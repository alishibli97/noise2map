from setuptools import setup, find_packages

setup(
    name="noise2map",
    version="1.0.0",
    author="Ali Shibli",
    author_email="shibli@kth.se",
    description="Noise2Map: End-to-End Diffusion Model for Semantic Segmentation and Change Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alishibli97/noise2map",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.27.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "rasterio>=1.3.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
