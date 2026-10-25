#!/usr/bin/env bash
# Run inside Ubuntu / WSL. Installs the actual transport engine and downloads
# its official physics and ICRP110 datasets. TORUS_GEANT4_PREBUILT=0 builds source.
set -euo pipefail
GEANT4_VERSION="11.4.3"
GEANT4_PREFIX="${GEANT4_PREFIX:-/opt/geant4}"
TORUS_PREFIX="${TORUS_PREFIX:-/opt/magnetic-torus}"
TASK_BUILD_ROOT="${TORUS_BUILD_ROOT:-$HOME/.cache/magnetic-torus-geant4}"
TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_BUILD_ROOT"
if [[ "${1:-}" != "--phantom-data" ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake ninja-build curl ca-certificates python3 libexpat1-dev zlib1g-dev nlohmann-json3-dev
  # Optional prebuilt environment avoids compiling several thousand toolkit
  # translation units. The release and all physics datasets are pinned by the
  # package specification; package URLs are captured in the explicit lock file.
  if [[ "${TORUS_GEANT4_PREBUILT:-1}" == "1" && ( ! -d "$GEANT4_PREFIX/conda-meta" || ! -f "$GEANT4_PREFIX/include/expat.h" ) ]]; then
    if [[ ! -x "$TASK_BUILD_ROOT/bin/micromamba" ]]; then
      curl --fail --location --retry 3 "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "$TASK_BUILD_ROOT/micromamba.tar.bz2"
      tar -xjf "$TASK_BUILD_ROOT/micromamba.tar.bz2" -C "$TASK_BUILD_ROOT" bin/micromamba
    fi
    TASK_MAMBA_ACTION="create"
    if [[ -d "$GEANT4_PREFIX/conda-meta" ]]; then TASK_MAMBA_ACTION="install"; fi
    if [[ -f "$TASK_REPO/env/geant4-linux-64.explicit.txt" ]]; then
      sudo env MAMBA_ROOT_PREFIX="$TASK_BUILD_ROOT/mamba" "$TASK_BUILD_ROOT/bin/micromamba" "$TASK_MAMBA_ACTION" -y \
        -p "$GEANT4_PREFIX" -f "$TASK_REPO/env/geant4-linux-64.explicit.txt"
    else
      sudo env MAMBA_ROOT_PREFIX="$TASK_BUILD_ROOT/mamba" "$TASK_BUILD_ROOT/bin/micromamba" "$TASK_MAMBA_ACTION" -y \
        -p "$GEANT4_PREFIX" -c conda-forge "geant4=${GEANT4_VERSION}=*noqt*" expat zlib freetype
    fi
  fi
  if [[ ! -f "$GEANT4_PREFIX/lib/cmake/Geant4/Geant4Config.cmake" && ! -f "$GEANT4_PREFIX/lib64/cmake/Geant4/Geant4Config.cmake" ]]; then
    if [[ ! -f "$TASK_BUILD_ROOT/geant4-${GEANT4_VERSION}.tar.gz" ]]; then
      curl --fail --location --retry 3 "https://gitlab.cern.ch/geant4/geant4/-/archive/v${GEANT4_VERSION}/geant4-v${GEANT4_VERSION}.tar.gz" -o "$TASK_BUILD_ROOT/geant4-${GEANT4_VERSION}.tar.gz"
    fi
    if [[ ! -f "$TASK_BUILD_ROOT/geant4-v${GEANT4_VERSION}/CMakeLists.txt" ]]; then
      tar -xzf "$TASK_BUILD_ROOT/geant4-${GEANT4_VERSION}.tar.gz" -C "$TASK_BUILD_ROOT"
    fi
    cmake -S "$TASK_BUILD_ROOT/geant4-v${GEANT4_VERSION}" -B "$TASK_BUILD_ROOT/geant4-build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$GEANT4_PREFIX" \
      -DGEANT4_INSTALL_DATA=ON -DGEANT4_BUILD_MULTITHREADED=OFF \
      -DGEANT4_USE_SYSTEM_EXPAT=ON -DGEANT4_USE_SYSTEM_ZLIB=ON
    cmake --build "$TASK_BUILD_ROOT/geant4-build" --parallel "${TORUS_BUILD_JOBS:-4}"
    sudo cmake --install "$TASK_BUILD_ROOT/geant4-build"
  fi
  if [[ -d "$GEANT4_PREFIX/conda-meta" ]]; then
    export CONDA_PREFIX="$GEANT4_PREFIX"
    export PATH="$GEANT4_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$GEANT4_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    set +u # Conda activation scripts deliberately inspect unset prior values.
    for TASK_ACTIVATE in "$GEANT4_PREFIX"/etc/conda/activate.d/*.sh; do
      if [[ -f "$TASK_ACTIVATE" ]]; then source "$TASK_ACTIVATE"; fi
    done
    set -u
    python3 "$TASK_REPO/magnetic_torus/geant4/export_lock.py" "$GEANT4_PREFIX" "$TASK_BUILD_ROOT/geant4-linux-64.explicit.txt"
  else
    source "$GEANT4_PREFIX/bin/geant4.sh"
  fi
  cmake -U 'EXPAT_*' -U 'ZLIB_*' -U 'FREETYPE_*' -S "$TASK_REPO/magnetic_torus/geant4" -B "$TASK_BUILD_ROOT/worker-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$GEANT4_PREFIX" -DCMAKE_INSTALL_PREFIX="$TORUS_PREFIX"
  cmake --build "$TASK_BUILD_ROOT/worker-build" --parallel "${TORUS_BUILD_JOBS:-4}"
  sudo cmake --install "$TASK_BUILD_ROOT/worker-build"
  # Both native and conda toolkits use this stable runtime entry point.
  TASK_WRAPPER="$TASK_BUILD_ROOT/torus-dose-run"
  cat > "$TASK_WRAPPER" <<EOF
#!/usr/bin/env bash
set -e
export CONDA_PREFIX="$GEANT4_PREFIX"
export PATH="$GEANT4_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="$GEANT4_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
if [[ -d "$GEANT4_PREFIX/conda-meta" ]]; then
  for TASK_ACTIVATE in "$GEANT4_PREFIX"/etc/conda/activate.d/*.sh; do
    if [[ -f "\$TASK_ACTIVATE" ]]; then source "\$TASK_ACTIVATE"; fi
  done
else
  source "$GEANT4_PREFIX/bin/geant4.sh"
fi
exec "$TORUS_PREFIX/bin/torus-dose" "\$@"
EOF
  sudo install -m 755 "$TASK_WRAPPER" "$TORUS_PREFIX/bin/torus-dose-run"
fi
if [[ ! -f "$TORUS_PREFIX/data/ICRPdata/MaleData.dat" ]]; then
  curl --fail --location --retry 3 \
    "https://cern.ch/geant4-data/datasets/examples/advanced/ICRP110Phantoms/ICRPdata.tar.gz" \
    -o "$TASK_BUILD_ROOT/ICRPdata.tar.gz"
  sudo mkdir -p "$TORUS_PREFIX/data"
  sudo tar -xzf "$TASK_BUILD_ROOT/ICRPdata.tar.gz" -C "$TORUS_PREFIX/data"
fi
printf '%s\n' "Geant4/ICRP110 installed under $GEANT4_PREFIX and $TORUS_PREFIX."
