# Legacy TensorFlow GPU container

The root Dockerfile packages the original TensorFlow/Tkinter Stefan application
from source and starts `interface3.py --legacy`. It uses Python 3.12 and the
TensorFlow 2.21 NVIDIA dependencies from `requirements-gpu.txt`. The modern
Qt/PyTorch workspace uses the separate desktop environment described in the main
README; its optional FEniCS and Geant4 workers are not part of this legacy image.

The image runs as UID/GID 10001. Application code and the Python environment are
root-owned, while `/data` and the user's home are writable. Mount a named volume
at `/data` to retain exported plots between runs. The build context contains only
the required legacy sources and verification helpers.

Closing the dashboard also closes its own embedded pyplot figure so that its
separate Tk manager cannot keep the container alive. Other callers' figures stay
open.

## Build

Run this from the repository root on a Linux x86-64 Docker host:

```sh
docker build --progress=plain -t heat-equation-legacy-gpu .
```

On Windows, NVIDIA passthrough requires Docker Desktop's WSL2 backend and a
compatible NVIDIA Windows driver. See [Docker's GPU prerequisites](https://docs.docker.com/desktop/features/gpu/).
The image's entrypoint exposes the CUDA libraries installed inside its own
Python environment; it does not install a host driver or change Docker settings.

## Verify the image

The GUI check creates and renders the actual Tkinter application under an isolated
Xvfb display. It also checks the runtime identity, writable output directory, and
protection of the source files:

Use `--init` to forward signals and reap child processes. It also lets
`xvfb-run` receive Xvfb's readiness signal; running that wrapper as PID 1 can stall.

```sh
docker run --rm --init --network none --cpus 2 --memory 2g --pids-limit 256 heat-equation-legacy-gpu xvfb-run -a --server-args="-screen 0 1600x1000x24 -nolisten tcp" python /opt/heat-equation/docker/legacy_container_smoke.py
```

The image also includes the legacy material-curve and figure-close regressions:

```sh
docker run --rm --init --network none --cpus 2 --memory 1g --pids-limit 128 -e MPLBACKEND=Agg heat-equation-legacy-gpu python -m unittest discover -s /opt/heat-equation/tests -p 'test_*.py'
```

The GPU case is opt-in and skipped in that headless unit run. Verify it separately
with the existing GPU training regression inside the same image:

```sh
docker run --rm --init --gpus all --network none --cpus 2 --memory 3g --pids-limit 256 -e RUN_LEGACY_TF_GPU_TEST=1 -e TF_FORCE_GPU_ALLOW_GROWTH=true heat-equation-legacy-gpu python -m unittest discover -s /opt/heat-equation/tests -p test_legacy_tensorflow_gpu.py
```

This opt-in test requires a detected GPU, a real PINN weight update, and GPU
backpropagation operations in a TensorFlow training trace. A skipped test or a
CPU-only result does not verify GPU support. GUI startup and one training epoch
are compatibility checks, rather than scientific qualification of a simulation.

## Interactive display and outputs

The default command opens a GUI and needs an already-authorized X11 display.
From a Linux or WSL shell with a working local X11 display, pass its `DISPLAY`
and mount only its socket directory:

```sh
docker run --rm --init --gpus all --cpus 2 --memory 3g --pids-limit 256 -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v heat-equation-results:/data heat-equation-legacy-gpu
```

If the display requires an Xauthority cookie, provide that display's cookie
read-only and set `XAUTHORITY` to its container path. Preserve the display's
access controls; a missing or unauthorized display is a setup failure. The Xvfb
check above verifies rendering independently of the host display configuration.
No image publication or deployment is part of these build and test commands.
