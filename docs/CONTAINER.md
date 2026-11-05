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

An empty named volume is initialized with the image's UID/GID 10001 ownership.
Pre-existing volumes and bind mounts retain their existing permissions and must
grant UID/GID 10001 write access before starting the application. Use a dedicated
writable volume or provision the desired ownership in your storage setup. Check
an existing volume with the same non-root user before opening the GUI:

```sh
docker run --rm --init --network none --cpus 1 --memory 128m --pids-limit 32 -v heat-equation-results:/data heat-equation-legacy-gpu python -c "import tempfile; probe = tempfile.TemporaryFile(dir='/data'); probe.write(b'write-check'); probe.close(); print('Output volume is writable')"
```

The first PNG export creates a unique `heat-equation-session-*` directory under
the output directory. Each App/session retains the latest two completed
`PINN_temperature_surface.png` and `Tx.png` snapshots there; repeated UI renders
replace those two files atomically. A failed render preserves the previous
successful snapshot. Concurrent App/container instances use separate directories.
Session directories remain after the application closes as persistent reference
images, so retaining a session does not save an image for every UI frame.
Exports use normal creation permissions filtered by the process umask: on Linux,
umask `022` gives directories `0755` and PNGs `0644`, allowing a host reader with
different UID/GID to access them when the mounted parent permits traversal. A
restrictive umask such as `077` intentionally retains owner-only access. Configure
the mount's ownership, traversal permissions and ACLs for the intended consumers;
the exporter does not override that access policy.

These reference images are intentionally retained across launches. The operator
chooses volume quotas and manual retention for the research outputs; the
application does not automatically delete prior sessions. An export I/O failure,
including exhausted storage, produces a warning while plot drawing continues.

Abrupt termination, such as a forced process kill or power loss during a write,
can leave an incomplete `.Tx-*.tmp` or `.PINN_temperature_surface-*.tmp` file.
Those temporary files are separate from the two completed snapshots. A new App
creates its own default session directory rather than resuming the interrupted
one. Include orphaned temporary files in the operator's quota/manual cleanup
policy, and remove them only after confirming that the session/writer is inactive.
The application does not scan or delete other sessions' files or active writers.

For a legacy single-writer consumer that needs stable filenames in a specific
directory, set the local operator option `HEAT_PNG_OUTPUT_DIR`. For example,
`HEAT_PNG_OUTPUT_DIR=.` writes `PINN_temperature_surface.png` and `Tx.png` in the
working directory. Relative paths resolve when the App first exports; the
selected directory is then retained for that App. In PowerShell:

```powershell
$env:HEAT_PNG_OUTPUT_DIR = "."
python interface3.py --legacy
```

For the container, pass `-e HEAT_PNG_OUTPUT_DIR=/data/legacy` to use that explicit
directory on the mounted output volume. Atomic replacement and failed-write
preservation still apply. Give each concurrent writer its own explicit directory;
sharing an override intentionally selects the legacy single-writer contract.
Leaving the option unset uses isolated session directories. No repository
production reader of the old working-directory PNGs was identified; this option
is a compatibility contract for external consumers.

Closing the dashboard also closes its own embedded pyplot figure so that its
separate Tk manager cannot keep the container alive. Other callers' figures stay
open.

## Build

Run this from the repository root on a Linux x86-64 Docker host:

```sh
docker build --progress=plain -t heat-equation-legacy-gpu .
```

The image uses its own Linux x86-64 / CPython 3.12 locks in `docker/`, including
the pip bootstrap. All 61 distributions retain the versions from the verified
GPU image; the shared application requirements are unchanged. Every permitted
wheel hash was taken from that version's official PyPI release JSON. Docker
uses only wheels and `--require-hashes` for both installation steps, so the cold
install verifies downloaded bytes rather than trusting version pins alone.
`docker/requirements-lock-provenance.json` records the inspected image ID,
inventory digest, original requirement-file byte digests, release-metadata
digests and permitted wheel filenames/URLs/hashes. Source-file line endings
affect those raw byte digests. Separate normalized UTF-8/LF input digests are
checked before pip runs, so CRLF and LF checkouts both work. A later change to
either shared requirement file fails the build with an instruction to regenerate
and verify the lock; a security update cannot silently leave the image lock stale.
The checker also verifies normalized digests of both copied lock files against
the recorded provenance before pip runs. A manual lock edit, missing lock, or
mixed generation therefore fails closed.

Generation stages all three files before replacing the two locks and then the
provenance commit record. This is not an atomic transaction across three files:
an interrupted or competing publication can leave a mixed set. Docker copies
the files into an immutable build snapshot, where the digest check rejects such
a set before installation; it never claims that mismatched files describe the
proven environment. Regenerate and verify a complete set after an interruption.

Before querying PyPI or publishing any output, the generator validates that the
captured inventory satisfies the two shared requirement files, their confined
local include, active PEP 508 markers/version constraints and requested extras
with their dependency closure. Markers use the captured Linux/x86-64/CPython 3.12
environment, including package `Requires-Dist` and `Provides-Extra` metadata;
the host's Windows/Python environment does not choose the dependencies. Unknown
pip options, other includes and direct/remote URLs are rejected with diagnostics
that omit their input contents. Run the generator with Python and pip present
(as in the verified image); it uses pip's vendored packaging parser without
installing another package. An operator-supplied inventory is evidence, not a
cryptographic attestation of the selected image.

The original `source_image`/`inventory_sha256` fields retain the pin-selection
origin. The `validation_source_image`/`validation_inventory_sha256` fields and
`requirements_validation` record the extended dependency/marker validation.
They may identify a later runtime-equivalent verified image without changing
the already proven lock bodies or package versions.

To regenerate after deliberately verifying another compatible image, capture
its distribution/dependency metadata, marker environment and target tags, then run the generator from this
repository. Review every version/hash change and repeat the cold image, GUI and
GPU checks. This pins Python wheels; apt repositories and the resulting image
bytes are not claimed to be reproducible from a historical OS-package snapshot.

```sh
docker run --rm --read-only --network none heat-equation-legacy-gpu python -I -c 'import importlib.metadata,json,sys; from pip._vendor.packaging.tags import sys_tags; from pip._vendor.packaging.markers import default_environment; print(json.dumps({"python":sys.version,"environment":default_environment(),"distributions":[{"name":d.metadata["Name"],"version":d.version,"requires_dist":d.metadata.get_all("Requires-Dist",[]),"provides_extra":d.metadata.get_all("Provides-Extra",[])} for d in importlib.metadata.distributions()],"tags":[str(t) for t in sys_tags()]}))' > proven-inventory.json
docker image inspect heat-equation-legacy-gpu --format '{{.Id}}'
python docker/generate_container_lock.py --inventory proven-inventory.json --source-image sha256:REPLACE_WITH_INSPECTED_IMAGE_ID --metadata-dir .cache/pypi-lock-metadata
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
