# This image runs the original TensorFlow/Tkinter application. The modern
# Qt/PyTorch desktop has its own requirements-desktop.txt environment.
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends -y \
      ca-certificates python3 python3-venv python3-tk libstdc++6 libgomp1 \
      fonts-dejavu-core xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/heat-equation
COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python -m pip install --no-cache-dir --only-binary=:all: -r requirements-gpu.txt \
    && /opt/venv/bin/python -m pip check \
    && groupadd --gid 10001 torus \
    && useradd --uid 10001 --gid 10001 --create-home torus \
    && install -d -o 10001 -g 10001 /data

COPY --chmod=0644 interface3.py backend.py deepLearning.py np_compat.py ./
COPY --chmod=0755 docker/legacy-entrypoint.sh /usr/local/bin/legacy-entrypoint
COPY --chmod=0644 docker/legacy_container_smoke.py ./docker/legacy_container_smoke.py
COPY --chmod=0644 tests/test_legacy_tensorflow_gpu.py ./tests/test_legacy_tensorflow_gpu.py
COPY --chmod=0644 tests/test_legacy_window_close.py tests/test_enthalpy_temperature_curve.py ./tests/
# COPY's file mode can also reach newly created destination directories.
# Keep these root-owned helper directories traversable by the runtime user.
RUN chmod 0755 /opt/heat-equation/docker /opt/heat-equation/tests

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/heat-equation \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/home/torus \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /data
USER 10001:10001
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/legacy-entrypoint"]
CMD ["python", "/opt/heat-equation/interface3.py", "--legacy"]
