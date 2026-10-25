#!/usr/bin/env bash
# Run inside Ubuntu / WSL. Installs the actual transport engine and downloads
# its official physics and ICRP110 datasets. TORUS_GEANT4_PREBUILT=0 builds source.
set -euo pipefail
GEANT4_VERSION="11.4.3"
MICROMAMBA_VERSION="2.3.3-0"
GEANT4_PREFIX="${GEANT4_PREFIX:-/opt/geant4}"
TORUS_PREFIX="${TORUS_PREFIX:-/opt/magnetic-torus}"
TASK_BUILD_ROOT="${TORUS_BUILD_ROOT:-$HOME/.cache/magnetic-torus-geant4}"
TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_BUILD_ROOT"
# Always consume a private, verified copy. Existing binaries, extracted sources,
# and build trees in the download cache are not trusted installation inputs.
TASK_STAGE="$(mktemp -d "$TASK_BUILD_ROOT/.setup.XXXXXXXX")"
trap 'rm -rf -- "$TASK_STAGE"' EXIT

verify_download() {
  local TASK_FILE="$1" TASK_NAME="$2" TASK_EXPECTED TASK_ACTUAL
  TASK_EXPECTED="$(awk -v name="$TASK_NAME" '{ sub(/\r$/, "") } $2 == name { print $1 }' "$TASK_REPO/env/geant4-downloads.sha256")"
  if [[ ! "$TASK_EXPECTED" =~ ^[0-9a-f]{64}$ || -L "$TASK_FILE" || ! -f "$TASK_FILE" ]]; then
    printf 'Invalid pinned download or checksum: %s\n' "$TASK_NAME" >&2
    return 1
  fi
  TASK_ACTUAL="$(sha256sum -- "$TASK_FILE")"
  if [[ "${TASK_ACTUAL%% *}" != "$TASK_EXPECTED" ]]; then
    printf 'SHA256 verification failed: %s\n' "$TASK_NAME" >&2
    return 1
  fi
}

verified_download() {
  local TASK_URL="$1" TASK_NAME="$2" TASK_CACHE="$TASK_BUILD_ROOT/$2"
  if [[ -L "$TASK_CACHE" || ( -e "$TASK_CACHE" && ! -f "$TASK_CACHE" ) ]]; then
    printf 'Invalid cached download: %s\n' "$TASK_NAME" >&2
    return 1
  fi
  if [[ ! -f "$TASK_CACHE" ]]; then
    curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
      "$TASK_URL" -o "$TASK_STAGE/$TASK_NAME.download"
    verify_download "$TASK_STAGE/$TASK_NAME.download" "$TASK_NAME"
    mv -fT -- "$TASK_STAGE/$TASK_NAME.download" "$TASK_CACHE"
  fi
  cp -- "$TASK_CACHE" "$TASK_STAGE/$TASK_NAME"
  verify_download "$TASK_STAGE/$TASK_NAME" "$TASK_NAME"
}

if [[ "${1:-}" != "--phantom-data" ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake ninja-build curl ca-certificates python3 libexpat1-dev zlib1g-dev nlohmann-json3-dev
  # Optional prebuilt environment avoids compiling several thousand toolkit
  # translation units. The release and all physics datasets are pinned by the
  # package specification; package URLs are captured in the explicit lock file.
  if [[ "${TORUS_GEANT4_PREBUILT:-1}" == "1" && ( ! -d "$GEANT4_PREFIX/conda-meta" || ! -f "$GEANT4_PREFIX/include/expat.h" ) ]]; then
    if [[ ! -f "$TASK_REPO/env/geant4-linux-64.explicit.txt" ]]; then
      printf 'Missing pinned Geant4 package lock: env/geant4-linux-64.explicit.txt\n' >&2
      exit 1
    fi
    verified_download "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64.tar.bz2" "micromamba-${MICROMAMBA_VERSION}.tar.bz2"
    mkdir "$TASK_STAGE/bootstrap"
    tar --no-same-owner --no-same-permissions -xjf "$TASK_STAGE/micromamba-${MICROMAMBA_VERSION}.tar.bz2" -C "$TASK_STAGE/bootstrap" bin/micromamba
    verify_download "$TASK_STAGE/bootstrap/bin/micromamba" "micromamba-${MICROMAMBA_VERSION}"
    TASK_MAMBA_ACTION="create"
    if [[ -d "$GEANT4_PREFIX/conda-meta" ]]; then TASK_MAMBA_ACTION="install"; fi
    sudo env MAMBA_ROOT_PREFIX="$TASK_BUILD_ROOT/mamba" "$TASK_STAGE/bootstrap/bin/micromamba" "$TASK_MAMBA_ACTION" -y \
      -p "$GEANT4_PREFIX" -f "$TASK_REPO/env/geant4-linux-64.explicit.txt"
  fi
  if [[ ! -f "$GEANT4_PREFIX/lib/cmake/Geant4/Geant4Config.cmake" && ! -f "$GEANT4_PREFIX/lib64/cmake/Geant4/Geant4Config.cmake" ]]; then
    verified_download "https://gitlab.cern.ch/geant4/geant4/-/archive/v${GEANT4_VERSION}/geant4-v${GEANT4_VERSION}.tar.gz" "geant4-${GEANT4_VERSION}.tar.gz"
    tar --no-same-owner --no-same-permissions -xzf "$TASK_STAGE/geant4-${GEANT4_VERSION}.tar.gz" -C "$TASK_STAGE"
    cmake -S "$TASK_STAGE/geant4-v${GEANT4_VERSION}" -B "$TASK_STAGE/geant4-build" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$GEANT4_PREFIX" \
      -DGEANT4_INSTALL_DATA=ON -DGEANT4_BUILD_MULTITHREADED=OFF \
      -DGEANT4_USE_SYSTEM_EXPAT=ON -DGEANT4_USE_SYSTEM_ZLIB=ON
    cmake --build "$TASK_STAGE/geant4-build" --parallel "${TORUS_BUILD_JOBS:-4}"
    sudo cmake --install "$TASK_STAGE/geant4-build"
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
  cmake -U 'EXPAT_*' -U 'ZLIB_*' -U 'FREETYPE_*' -S "$TASK_REPO/magnetic_torus/geant4" -B "$TASK_STAGE/worker-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$GEANT4_PREFIX" -DCMAKE_INSTALL_PREFIX="$TORUS_PREFIX"
  cmake --build "$TASK_STAGE/worker-build" --parallel "${TORUS_BUILD_JOBS:-4}"
  sudo cmake --install "$TASK_STAGE/worker-build"
  # Both native and conda toolkits use this stable runtime entry point.
  TASK_WRAPPER="$TASK_STAGE/torus-dose-run"
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
  verified_download "https://cern.ch/geant4-data/datasets/examples/advanced/ICRP110Phantoms/ICRPdata.tar.gz" "ICRPdata.tar.gz"
  mkdir "$TASK_STAGE/phantom"
  tar --no-same-owner --no-same-permissions -xzf "$TASK_STAGE/ICRPdata.tar.gz" -C "$TASK_STAGE/phantom"
  sudo mkdir -p "$TORUS_PREFIX/data"
  sudo cp -R -- "$TASK_STAGE/phantom/ICRPdata" "$TORUS_PREFIX/data/"
fi
printf '%s\n' "Geant4/ICRP110 installed under $GEANT4_PREFIX and $TORUS_PREFIX."
