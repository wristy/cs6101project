#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

prepare_source() {
    local name="$1" url="$2" revision="$3"
    local destination="external/$name" patch="$PWD/patches/$name.patch"
    if [[ ! -e "$destination" ]]; then
        git clone "$url" "$destination"
        git -C "$destination" checkout --detach "$revision"
    fi
    if [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
        printf 'Unexpected revision in %s; leaving it unchanged.\n' "$destination" >&2
        return 1
    fi
    if git -C "$destination" apply --reverse --check "$patch" 2>/dev/null; then
        printf 'Compatibility patch already present: %s\n' "$name"
    else
        git -C "$destination" apply --check "$patch"
        git -C "$destination" apply "$patch"
    fi
}

prepare_source thought-tracing https://github.com/skywalker023/thought-tracing.git 966eb2712714aa29b4d5d2395e39af94a2be90d7
prepare_source AutoToM https://github.com/SCAI-JHU/AutoToM.git 3f569b7ab1d0ee2702ab43ebdc724679d5fc5231
uv sync --extra dev --frozen
.venv/bin/python -m ipykernel install --user --name tom-research --display-name 'ToM Research (.venv)'
.venv/bin/python -m tom_research.study prepare
.venv/bin/python -m pytest tests -q