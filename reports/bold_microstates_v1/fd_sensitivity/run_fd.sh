#!/usr/bin/env bash
set -euo pipefail
umask 077
work=$(cd -- "$(dirname -- "$0")" && pwd -P)
original=/data1/student/ruidong/bold_microstates_v1_20261121_Jp2fwhug
syn9=/data1/student/ruidong/weida_stn_own_20261116_j_srnj4i
image=sha256:92a4a0ac1d2be762484ce11c0de2183718c517d8d91b59ef942ea6095e309183
trap 'rc=$?; printf "%s\n" "$rc" > "$work/job.exit"; date -u +%FT%TZ; exit "$rc"' EXIT
date -u +%FT%TZ
test "$(git get-tar-commit-id < "$work/experiment.tar")" = 540c114deb03309f347e62d3cd0bb7912305b66e
mkdir "$work/repo"
tar -xf "$work/experiment.tar" -C "$work/repo"
run_python() {
  docker run --rm --pull=never --network=none --read-only \
    --user "$(id -u):$(id -g)" --cap-drop=ALL --security-opt=no-new-privileges \
    --tmpfs /tmp:rw,nosuid,nodev,size=536870912 \
    --mount "type=bind,src=$syn9,dst=$syn9,readonly" \
    --mount "type=bind,src=$original,dst=$original,readonly" \
    --mount "type=bind,src=$work,dst=$work" --workdir "$work/repo" \
    --env HOME=/tmp --env MPLCONFIGDIR=/tmp/mpl --env PYTHONDONTWRITEBYTECODE=1 \
    --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
    --env BLIS_NUM_THREADS=1 --entrypoint /app/.pixi/envs/xcp-d/bin/python "$image" "$@"
}
run_python -m unittest discover -s experiments/bold_microstates_v1/tests -v > "$work/tests.log" 2>&1
tail -n 4 "$work/tests.log"
run_python experiments/bold_microstates_v1/fd_sensitivity.py --run-dir "$original" --out "$work/summary"
cat "$work/summary/fd_adjusted_summary.json"
