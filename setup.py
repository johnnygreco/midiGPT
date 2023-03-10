import os

from setuptools import find_packages, setup

if os.path.isfile("VERSION"):
    with open("VERSION") as version:
        __version__ = version.read().strip("v")
else:
    __version__ = "0.0.0"

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
)
