#!/usr/bin/env bash
# install.sh — mindv environment installer
# Run from the root of your minDV repo:  bash install.sh
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="mindv"
ENV_FILE="environment.yml"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'
BLU='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${BLU}▶${NC} $*"; }
ok()    { echo -e "${GRN}✔${NC} $*"; }
warn()  { echo -e "${YEL}⚠${NC} $*"; }
die()   { echo -e "${RED}✗ ERROR:${NC} $*" >&2; exit 1; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║          mindv — environment installer              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Find conda base ────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null) \
    || die "conda not found. Install miniforge3 first."
info "conda base: $CONDA_BASE"

# ── 2. Prefer mamba, fall back to conda ──────────────────────────
if command -v mamba &>/dev/null; then
    SOLVER="mamba"; info "Using mamba (faster solver)"
else
    SOLVER="conda"; warn "mamba not found — using conda (slower)"
fi

# ── 3. Verify environment.yml exists ─────────────────────────────
[[ -f "$ENV_FILE" ]] || die "$ENV_FILE not found. Run from the minDV repo root."

# ── 4. Remove old environment if requested ────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists."
    read -rp "    Remove and recreate? [y/N] " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        info "Removing old environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        info "Updating existing environment instead..."
        $SOLVER env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
        SKIP_CREATE=1
    fi
fi

# ── 5. Create the environment ─────────────────────────────────────
if [[ -z "${SKIP_CREATE:-}" ]]; then
    info "Creating environment '${ENV_NAME}'..."
    info "(This takes 5-15 minutes on first install)"
    echo ""

    # Use flexible channel priority to avoid pytorch/torchvision conflict
    $SOLVER env create \
        -n "$ENV_NAME" \
        -f "$ENV_FILE" \
        --channel-priority flexible \
        || die "Environment creation failed. See messages above."
fi

# ── 6. Install mindv itself in editable mode ─────────────────────
info "Installing mindv package (editable)..."
conda run -n "$ENV_NAME" pip install -e "$REPO_DIR" --quiet \
    || die "pip install -e failed. Check setup.py."

# ── 7. Update TBProfiler database ────────────────────────────────
info "Updating TBProfiler resistance database..."
conda run -n "$ENV_NAME" tb-profiler update_tbdb \
    && ok "TBProfiler database updated." \
    || warn "TBProfiler update failed (no internet?). Run manually: conda run -n mindv tb-profiler update_tbdb"

# ── 8. Smoke test ─────────────────────────────────────────────────
echo ""
info "Smoke tests..."
conda run -n "$ENV_NAME" python - << 'PYEOF'
import torch, pysam, Bio, numpy, pandas, sklearn
print(f"  torch      {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
print(f"  pysam      {pysam.__version__}")
print(f"  biopython  {Bio.__version__}")
print(f"  numpy      {numpy.__version__}")
print(f"  pandas     {pandas.__version__}")
import subprocess
for tool in ["samtools", "bwa", "fastp", "gatk", "tb-profiler"]:
    r = subprocess.run([tool, "--version"], capture_output=True, text=True)
    ver = (r.stdout or r.stderr).splitlines()[0][:60]
    print(f"  {tool:<14} {ver}")
PYEOF

# ── 9. Test mindv CLI ─────────────────────────────────────────────
conda run -n "$ENV_NAME" mindv --version \
    && ok "mindv CLI working." \
    || warn "mindv CLI not found — check setup.py entry_points."

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GRN}Installation complete!${NC}"
echo ""
echo "  Activate:     conda activate mindv"
echo "  Test model:   mindv test"
echo "  Build panel:  mindv panel-build --help"
echo "  Run scan:     mindv scan --help"
echo ""
echo "  To deactivate: conda deactivate"
echo ""
