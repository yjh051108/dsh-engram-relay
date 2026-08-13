#!/bin/bash
# Build the dsh-engram-relay external plugin: compile src/ → lib/ (JS) and
# lib/types/ (declarations) with the dsh checkout's TypeScript. Dependency
# resolution mirrors dsh-vision/session-chatlog: the plugin's node_modules
# holds symlinks into the dsh checkout, so tsc type-checks against the same
# vendored/workspace packages the running dsh ships. Requires `dsh` on PATH
# or DSH_CHECKOUT pointing at a dsh source checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKOUT="${DSH_CHECKOUT:-}"
if [ -z "$CHECKOUT" ]; then
  if command -v dsh &>/dev/null; then
    DSH_BIN=$(readlink -f "$(command -v dsh)" 2>/dev/null || command -v dsh)
    CHECKOUT=$(cd "$(dirname "$DSH_BIN")/../../.." && pwd)
  fi
fi
if [ -z "$CHECKOUT" ] || [ ! -d "$CHECKOUT/packages" ]; then
  echo "build: cannot locate the dsh checkout (set DSH_CHECKOUT or put dsh on PATH)" >&2
  exit 1
fi

TSC="$CHECKOUT/node_modules/.bin/tsc"
if [ ! -x "$TSC" ]; then
  echo "build: tsc not found at $TSC" >&2
  exit 1
fi

link_pkg() {
  local target="$CHECKOUT/$2"
  if [ ! -e "$target" ]; then
    echo "build: dependency target missing: $target" >&2
    exit 1
  fi
  # git-bash 的 rm/ln 在 Windows 上对目录不可靠，统一用 node 删除 +
  # junction 链接（无需管理员权限，与 Windows DSH 生态一致）。
  node -e "
    const fs = require('fs');
    const path = require('path');
    const link = path.resolve(process.argv[1]);
    const target = path.resolve(process.argv[2]);
    fs.rmSync(link, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(link), { recursive: true });
    fs.symlinkSync(target, link, process.platform === 'win32' ? 'junction' : 'dir');
  " "node_modules/$1" "$target"
}

echo "=== Linking build dependencies (checkout: $CHECKOUT) ==="
mkdir -p node_modules/@deepseek-ai
node -e "const fs=require('fs');fs.rmSync('node_modules/@standard-schema',{recursive:true,force:true})"
link_pkg cordis vendor/cordis
link_pkg cosmokit vendor/cosmokit
link_pkg schemastery vendor/schemastery
link_pkg @deepseek-ai/dsh-brand packages/util/brand
link_pkg @deepseek-ai/dsh-llm packages/llm/llm
link_pkg @deepseek-ai/dsh-compaction packages/compaction/compaction
link_pkg @deepseek-ai/dsh-scope packages/core/scope
link_pkg @deepseek-ai/dsh-session packages/core/session
link_pkg @deepseek-ai/dsh-system-prompt packages/core/system-prompt
link_pkg @deepseek-ai/dsh-tools packages/core/tools

# @standard-schema/spec: external npm types referenced by cordis/schemastery
# declarations, hoisted only inside the pnpm store.
STD_SCHEMA=$(find "$CHECKOUT/node_modules/.pnpm" -maxdepth 1 -type d -iname '@standard-schema+spec@*' 2>/dev/null | head -1)
if [ -n "$STD_SCHEMA" ]; then
  node -e "
    const fs = require('fs');
    const path = require('path');
    fs.rmSync('node_modules/@standard-schema', { recursive: true, force: true });
    fs.mkdirSync('node_modules/@standard-schema', { recursive: true });
    fs.symlinkSync(path.resolve(process.argv[1]), path.resolve('node_modules/@standard-schema/spec'), process.platform === 'win32' ? 'junction' : 'dir');
  " "$STD_SCHEMA/node_modules/@standard-schema/spec"
else
  echo "build: @standard-schema/spec not found in pnpm store; skipLibCheck may still cover it" >&2
fi

echo "=== Compiling src → lib (tsc $("$TSC" --version)) ==="
"$TSC" -p tsconfig.json

echo "=== Build complete ==="
ls -la lib/ lib/types/ 2>/dev/null
