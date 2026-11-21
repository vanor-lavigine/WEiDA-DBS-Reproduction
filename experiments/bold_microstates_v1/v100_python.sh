#!/usr/bin/env bash
# Invoke Python from the locally present XCP-D image without running preprocessing.
set -euo pipefail

run_dir=$1
source_root=$2
image_id=$3
shift 3

exec docker run --rm --pull=never --network=none --read-only \
  --user "$(id -u):$(id -g)" --cap-drop=ALL --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,nosuid,nodev,size=536870912 \
  --mount "type=bind,src=$source_root,dst=$source_root,readonly" \
  --mount "type=bind,src=$run_dir,dst=$run_dir" --workdir "$run_dir/repo" \
  --env HOME=/tmp --env MPLCONFIGDIR=/tmp/mpl --env PYTHONDONTWRITEBYTECODE=1 \
  --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OMP_NUM_THREADS=1 --env BLIS_NUM_THREADS=1 \
  --entrypoint /app/.pixi/envs/xcp-d/bin/python "$image_id" "$@"
