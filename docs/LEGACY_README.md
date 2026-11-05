# Legacy 2-Phase Stefan Problem Solver using PINNs

## Overview

This project involves a standalone Python 3 desktop application designed to solve the 2-phase Stefan problem using Physics-Informed Neural Networks (PINNs) along with analytical and numerical methods. The application integrates TensorFlow, Keras, CUDA for GPU acceleration, and Tkinter for the user interface.

## Author

Luba Ira - [lubacareer@gmail.com](mailto:lubacareer@gmail.com) - [irakorla@post.bgu.ac.il](mailto:irakorla@post.bgu.ac.il)

## Project Description

The application simulates the latent heat thermal energy storage in lunar regolith, which is crucial for future lunar settlement. The PINN approach allows for efficient and accurate predictions of phase change materials (PCMs) behavior under varying thermal conditions.

## Features

- **Physics-Informed Neural Networks (PINNs):** Dual-branch network architecture for temperature and boundary predictions.
- **Numerical Methods:** Explicit and implicit numerical solutions for handling the Stefan problem.
- **Analytical Solutions:** Provides baseline solutions for comparison.
- **Graphical User Interface (GUI):** Built with Tkinter for easy interaction.
- **GPU Acceleration:** Utilizes CUDA for computational efficiency.

## Files

- **`interface3.py`:** The main file containing the GUI and the application logic.
- **`deepLearning.py`:** Contains the PINN model architecture and training functions.
- **`backend.py`:** Includes utility functions and additional computational methods.

### How to Run the Program

Use the local setup below, or build the source-based [legacy NVIDIA GPU container](CONTAINER.md).
The previously published image is not used by that build.

## Local setup and NVIDIA GPU on WSL2

The legacy application uses Python 3.12 and TensorFlow 2.21. On Windows, native
TensorFlow runs on the CPU. For NVIDIA GPU acceleration, run the legacy GUI in
Ubuntu on WSL2 with a current NVIDIA Windows driver and WSLg. Install Conda in
Ubuntu, then from the repository root inside Ubuntu run:

```bash
conda create -n legacy-tf python=3.12 tk -y
conda activate legacy-tf
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
# Add the CUDA libraries installed in this environment to the loader path.
export LD_LIBRARY_PATH="$(python -c 'import glob, site; print(":".join(p for s in site.getsitepackages() for p in glob.glob(s + "/nvidia/*/lib")))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python interface3.py --legacy
```

Confirm that the device list contains a GPU before starting training. The GPU
requirements use TensorFlow's Linux CUDA dependencies; a Windows DirectML
plugin is not required. WSLg supplies the display for the Tkinter window. To
repeat the GPU training smoke test from the repository root, run:

```bash
RUN_LEGACY_TF_GPU_TEST=1 python -m unittest discover -s tests -p test_legacy_tensorflow_gpu.py
```

For CPU use on Windows or another supported platform, create a clean Python
3.12 environment, install `requirements.txt`, and run `python interface3.py
--legacy`. If NumPy fails to import, recreate the environment rather than
mixing incompatible Conda and pip packages.
