"""
Setup script for Audio Transcription Service
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="audio-transcription-service",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time audio transcription service with webhook integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-transcription-service",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "soundcard>=0.4.2",
        "pyaudio>=0.2.11",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "openai-whisper>=20230918",
        "webrtcvad>=2.0.10",
        "pydub>=0.25.1",
        "pystray>=0.19.4",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.5",
        "python-dotenv>=1.0.0",
        "psutil>=5.9.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "audio-transcription=system_tray_app:main",
            "audio-transcription-cli=cli:main",
        ],
        "gui_scripts": [
            "audio-transcription-gui=system_tray_app:main",
        ],
    },
    package_data={
        "": ["*.png", "*.ico", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)
