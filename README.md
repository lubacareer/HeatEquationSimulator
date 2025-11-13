# 2-Phase Stefan Problem Solver using PINNs

## Overview

This project involves a standalone Python 3 desktop application designed to solve the 2-phase Stefan problem using Physics-Informed Neural Networks (PINNs) along with analytical and numerical methods. The application integrates TensorFlow, Keras, CUDA for GPU acceleration, and Tkinter for the user interface.

## Author

Luba Ira - [lubani@ac.sce.ac.il](mailto:lubani@ac.sce.ac.il)

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

1. **Ensure Prerequisites Are Met:**
   - **Docker Desktop**: Make sure Docker Desktop is installed and running on your system.
   - **Xming (Windows users)**: Install and run Xming to display GUI applications. Configure it to allow external connections.
   - **X Server for Linux/macOS**: Ensure X11 (Linux) or XQuartz (macOS) is installed and running for GUI display.

2. **Pull the Docker image from Google Cloud Container Registry:**
    
    ```commandline
   docker pull gcr.io/sonic-cumulus-357519/scientific-project
    ```
3. **Run the Docker container:**

    ```commandline
   docker run -it --rm -e DISPLAY=host.docker.internal:0.0 gcr.io/sonic-cumulus-357519/scientific-project
    ```
   
4. **Display Configuration:**
   - Ensure the DISPLAY environment variable is correctly set for your operating system to connect to the X server (e.g., DISPLAY=host.docker.internal:0.0 for Windows users).

## Troubleshooting: NumPy DLL load failed

If you see an error similar to the following when launching the app or importing matplotlib/NumPy:

- ImportError: DLL load failed while importing _multiarray_umath: The specified module could not be found.
- ImportError: numpy._core.multiarray failed to import

This is an environment issue: your Python environment has an incompatible or broken NumPy build (often caused by mixing conda and pip, or installing wheels that do not match your Python version/ABI).

The application now detects this early and shows a friendly message before heavy imports. To fix your environment, choose ONE of the options below and run the commands in a clean environment.

Option A — Conda (recommended if you created a conda env):

    # Create or repair environment (example name: tf_env)
    conda create -n tf_env python==3.10 -y
    conda activate tf_env
    # Install compatible scientific stack from conda-forge
    conda install -c conda-forge "numpy==1.23.5" "scipy==1.10.*" "matplotlib" "scikit-learn>=1.5,<2" -y
    # Optional: TensorFlow CPU and DirectML plugin on Windows
    pip install --upgrade pip
    pip install --extra-index-url https://pkgs.dev.azure.com/azure-public/vside/_packaging/tensorflow-directml/pypi/simple/ \
        "tensorflow-cpu==2.10.0" "tensorflow-directml-plugin==0.4.0.dev230202"

Option B — Pure pip (no conda), in a clean venv:

    py -3.10 -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip uninstall -y numpy scipy
    pip install --only-binary=:all: "numpy==1.26.4" "scipy==1.11.4" matplotlib "scikit-learn>=1.5,<2"

Notes:
- Do NOT mix conda and pip for NumPy/SciPy in the same environment.
- Ensure your Python version matches the pinned versions in requirements.txt.
- On Python 3.12+, NumPy 2.x will be installed and our code handles NumPy 2 changes.

Local run (without Docker):

    python -m pip install -r requirements.txt
    python interface3.py

If the GUI exits early with a "NumPy import error" dialog, follow the steps above to repair your environment.
