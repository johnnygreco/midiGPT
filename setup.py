from setuptools import find_packages, setup

__version__ = "0.0.1-beta.5"

with open("README.md", "r") as fh:
    long_description = fh.read() or ""

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="midigpt",
    version=__version__,
    description="midiGPT: A MIDI Music Generation Library",
    long_description=long_description,
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    url="https://github.com/johnnygreco/midiGPT",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
