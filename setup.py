"""
Setup script for My2ndEye - Advanced Video Surveillance & AI Detection System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt" 
requirements = []
if requirements_path.exists():
    requirements = [line.strip() for line in requirements_path.read_text().splitlines()
                   if line.strip() and not line.startswith('#')]

setup(
    name="my2ndeye",
    version="1.0.0",
    author="My2ndEye Team",
    author_email="contact@my2ndeye.local",
    description="Advanced Video Surveillance & AI Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/my2ndeye/my2ndeye",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "my2ndeye.web": ["templates/*.html"],
        "my2ndeye.capture": ["*.c", "Makefile"],
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "my2ndeye-detect=my2ndeye.cli.detect:main",
            "my2ndeye-caption=my2ndeye.cli.caption:main", 
            "my2ndeye-serve=my2ndeye.cli.serve:main",
            "my2ndeye=my2ndeye.cli.detect:main",  # Default to detection
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    keywords="video surveillance ai detection yolo computer-vision security camera",
)