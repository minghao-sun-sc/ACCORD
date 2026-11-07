#!/usr/bin/env bash
# Build the EternaFold/CONTRAfold binary used by ACCORD.
#
# ACCORD runs both learned oracles with the `contrafold` binary from the EternaFold
# source tree at a pinned commit: with EternaFoldParams.v1 for EternaFold and with
# the built-in defaults for CONTRAfold. Requires git, make, perl and a C++ compiler
# (taken from $CXX if set, otherwise g++); the build uses x86 SSE flags, so it targets
# Linux x86_64.
#
# Usage: bash scripts/install_eternafold.sh [TARGET_DIR]   (default: ./external/EternaFold)
set -euo pipefail

COMMIT=13d24878efc2bd433535eb233379df15e2a7bed6
DEST=${1:-external/EternaFold}

if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/eternagame/EternaFold.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT" 2>/dev/null || {
    git -C "$DEST" fetch --quiet origin
    git -C "$DEST" checkout --quiet "$COMMIT"
}
make -C "$DEST/src" CXX="${CXX:-g++}" >/dev/null

HOME_DIR=$(cd "$DEST" && pwd)
test -x "$HOME_DIR/src/contrafold"
echo "Built $HOME_DIR/src/contrafold at commit $COMMIT."
echo "Add this to your shell (or conda env) so that ACCORD finds it:"
echo "  export ACCORD_ETERNAFOLD_HOME=$HOME_DIR"
