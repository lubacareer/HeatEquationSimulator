#!/usr/bin/env bash
# Run as root in Ubuntu WSL: sudo bash /mnt/c/.../scripts/setup_wsl_physics.sh
set -euo pipefail
if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3-dolfinx python3-gmsh python3-scipy cmake g++ nlohmann-json3-dev libgl1-mesa-dev libglu1-mesa-dev
/usr/bin/python3 - <<'PY'
import dolfinx, gmsh, petsc4py, scipy
print('DOLFINx', dolfinx.__version__, 'Gmsh', gmsh.__version__, 'SciPy', scipy.__version__)
print('FEniCS worker dependencies are ready. Geant4 is built separately.')
PY
lock_dir="${XDG_DATA_HOME:-$HOME/.local/share}/magnetic_torus"
mkdir -p "$lock_dir"
dpkg-query -W -f='${Package}=${Version}\n' python3-dolfinx python3-dolfinx-real python3-gmsh python3-petsc4py-real python3-basix python3-ufl python3-ffcx python3-numpy python3-scipy > "$lock_dir/linux-physics.lock"
printf 'Installed Linux physics package lock: %s\n' "$lock_dir/linux-physics.lock"
