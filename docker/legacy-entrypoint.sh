#!/bin/sh
set -eu

# TensorFlow's CUDA extra installs the runtime libraries inside this venv.
# Resolve those trusted package directories without importing TensorFlow or
# starting a GPU context before the application or its verification command.
# Isolated mode prevents a sysconfig.py in the writable /data volume (or an
# inherited PYTHONPATH) from changing this trusted package-root lookup.
nvidia_package_root=$(/opt/venv/bin/python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia")')
nvidia_library_path=
nvidia_binary_path=
for nvidia_component_directory in "$nvidia_package_root"/*/lib; do
    [ -d "$nvidia_component_directory" ] || continue
    nvidia_library_path="${nvidia_library_path:+$nvidia_library_path:}$nvidia_component_directory"
done
for nvidia_component_directory in "$nvidia_package_root"/*/bin; do
    [ -d "$nvidia_component_directory" ] || continue
    nvidia_binary_path="${nvidia_binary_path:+$nvidia_binary_path:}$nvidia_component_directory"
done
if [ -n "$nvidia_library_path" ]; then
    export LD_LIBRARY_PATH="$nvidia_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -n "$nvidia_binary_path" ]; then
    export PATH="$nvidia_binary_path:$PATH"
fi

exec "$@"
