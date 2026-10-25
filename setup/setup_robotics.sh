#!/usr/bin/env bash
#
# setup_robotics.sh — Physical-AI / robotics stack for Ubuntu 24.04 (Noble)
# Target: Intel Core Ultra (Meteor Lake), Arc iGPU, no NVIDIA GPU.
#
# Modules (run independently or together):
#   base   System build tools, Python venv tooling, common libs
#   argos  ARGoS3 swarm simulator + Buzz language (built from source)
#   pyenv  Python venv: MuJoCo, PyBullet, Gymnasium, Stable-Baselines3,
#          Ultralytics/YOLOv11, OpenVINO, imageio
#   edge   Intel Arc iGPU compute runtime + Intel NPU .deb driver (OpenVINO)
#   verify Smoke checks for everything above
#
# USAGE
#   chmod +x setup_robotics.sh
#   ./setup_robotics.sh base pyenv edge      # typical
#   ./setup_robotics.sh argos                # add the swarm sim
#   ./setup_robotics.sh verify
#
# Notes
#   - Run as your normal user (NOT root); it calls sudo only where needed.
#   - Re-running is safe: apt is idempotent, source repos are rebuilt in place.
#   - No CUDA by design (Intel iGPU). Isaac Sim/Lab + large GPU RL belong elsewhere.

set -uo pipefail

VENV_DIR="${HOME}/pai-robotics-venv"
SRC_DIR="${HOME}/pai-robotics-src"

c_blue=$'\e[1;34m'; c_green=$'\e[1;32m'; c_yellow=$'\e[1;33m'; c_red=$'\e[1;31m'; c_off=$'\e[0m'
section() { printf '\n%s==== %s ====%s\n' "$c_blue" "$1" "$c_off"; }
ok()   { printf '%s[ok]%s %s\n' "$c_green"  "$c_off" "$1"; }
warn() { printf '%s[!!]%s %s\n' "$c_yellow" "$c_off" "$1"; }
err()  { printf '%s[xx]%s %s\n' "$c_red"    "$c_off" "$1" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }

require_noble() {
  local c; c=$(. /etc/os-release && echo "$VERSION_CODENAME")
  [ "$c" = "noble" ] || warn "Expected Ubuntu 24.04 (noble); found '$c'. Continuing."
}

# ----- base ------------------------------------------------------------------
install_base() {
  section "base — build tools & common libraries"
  require_noble
  sudo apt update
  sudo apt install -y \
    build-essential cmake git curl wget pkg-config ca-certificates gnupg lsb-release \
    software-properties-common \
    python3 python3-pip python3-venv python3-dev \
    libeigen3-dev libopencv-dev
  mkdir -p "$SRC_DIR"
  ok "base packages installed"
}

# ----- python env ------------------------------------------------------------
install_pyenv() {
  section "pyenv — venv (MuJoCo, PyBullet, SB3, YOLOv11, OpenVINO)"
  sudo apt install -y python3-venv python3-pip python3-dev
  [ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip wheel
  pip install \
    numpy opencv-python \
    mujoco pybullet gymnasium "stable-baselines3[extra]" \
    imageio imageio-ffmpeg \
    ultralytics openvino
  deactivate
  ok "venv ready at ${VENV_DIR}"
  warn "Activate per session:  source ${VENV_DIR}/bin/activate"
  warn "Optional lean CPU torch: pip uninstall -y torch torchvision && \\"
  warn "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
}

# ----- edge AI accelerators --------------------------------------------------
install_edge() {
  section "edge — Arc iGPU compute runtime + Intel NPU .deb driver"

  # Arc iGPU compute runtime (kobuk-team intel-graphics PPA) for OpenVINO GPU plugin.
  sudo add-apt-repository -y ppa:kobuk-team/intel-graphics || warn "intel-graphics PPA failed"
  sudo apt update
  sudo apt install -y intel-opencl-icd libze1 libze-intel-gpu1 clinfo \
    || warn "iGPU compute runtime install incomplete"

  # Intel NPU user-space driver via the SYSTEM .deb (not the confined snap —
  # the snap is invisible to pip-installed OpenVINO's level-zero loader).
  sudo apt install -y libtbb12
  local tmp; tmp="$(mktemp -d)"
  ( cd "$tmp"
    local urls
    urls=$(curl -s https://api.github.com/repos/intel/linux-npu-driver/releases/latest \
           | grep -oP '"browser_download_url":\s*"\K[^"]*ubuntu2404[^"]*')
    if [ -z "$urls" ]; then warn "Could not resolve NPU driver assets (GitHub API)."; return 0; fi
    echo "NPU driver assets:"; echo "$urls"
    for u in $urls; do wget -q --show-progress "$u"; done
    for t in *.tar.gz; do [ -e "$t" ] && tar -xf "$t"; done
    sudo dpkg -i $(find . -name '*.deb' ! -name '*dbgsym*') 2>/dev/null; sudo apt -f install -y
  )
  rm -rf "$tmp"

  sudo usermod -aG render,video "$USER" || true
  ok "Edge accelerators set up."
  warn "Log out/in (or reboot) so the render/video group takes effect in every shell."
  warn "Verify: source ${VENV_DIR}/bin/activate && python -c \"import openvino as ov; print(ov.Core().available_devices)\""
}

# ----- argos + buzz ----------------------------------------------------------
install_argos() {
  section "argos — ARGoS3 + Buzz (from source)"
  sudo apt install -y \
    cmake libfreeimage-dev libfreeimageplus-dev \
    qtbase5-dev freeglut3-dev libxi-dev libxmu-dev \
    liblua5.3-dev lua5.3 doxygen graphviz libgraphviz-dev \
    asciidoc xmlto docbook-xml docbook-xsl xsltproc libxml2-utils
  mkdir -p "$SRC_DIR"

  # ARGoS3 — docs are disabled to avoid the doc-artifact install failures.
  if [ ! -d "${SRC_DIR}/argos3" ]; then
    git clone https://github.com/ilpincy/argos3.git "${SRC_DIR}/argos3"
  else
    git -C "${SRC_DIR}/argos3" pull --ff-only || warn "argos3 pull skipped"
  fi
  mkdir -p "${SRC_DIR}/argos3/build"
  ( cd "${SRC_DIR}/argos3/build" \
    && cmake -DCMAKE_BUILD_TYPE=Release -DARGOS_DOCUMENTATION=OFF ../src \
    && make -j"$(nproc)" \
    && sudo make install ) || { err "ARGoS3 build failed"; return 1; }
  sudo ldconfig
  ok "ARGoS3 installed ($(argos3 --version 2>/dev/null | head -n1 || echo installed))"

  # Buzz (links against the installed ARGoS, so it must go second).
  if [ ! -d "${SRC_DIR}/buzz" ]; then
    git clone https://github.com/buzz-lang/Buzz.git "${SRC_DIR}/buzz"
  else
    git -C "${SRC_DIR}/buzz" pull --ff-only || warn "buzz pull skipped"
  fi
  mkdir -p "${SRC_DIR}/buzz/build"
  ( cd "${SRC_DIR}/buzz/build" \
    && cmake ../src \
    && make -j"$(nproc)" \
    && sudo make install ) || { err "Buzz build failed"; return 1; }
  sudo ldconfig
  ok "Buzz installed. Compile scripts with: bzzc -I /usr/local/share/buzz/include ..."
}

# ----- verify ----------------------------------------------------------------
verify_all() {
  section "verify — smoke checks"
  have argos3 && ok "ARGoS: present ($(argos3 --version 2>/dev/null | head -n1))" || warn "ARGoS not found"
  have bzzc   && ok "Buzz: bzzc present" || warn "Buzz not found"
  if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    python - <<'PY' || warn "Python stack check hit an error"
import importlib
for m in ["mujoco","pybullet","gymnasium","stable_baselines3","ultralytics","openvino"]:
    try: importlib.import_module(m); print(f"  [ok] {m}")
    except Exception as e: print(f"  [xx] {m}: {e}")
try:
    import openvino as ov
    print("  OpenVINO devices:", ov.Core().available_devices)  # expect CPU, GPU, NPU
except Exception as e:
    print("  [xx] OpenVINO:", e)
PY
    deactivate
  else
    warn "venv not found at ${VENV_DIR}"
  fi
}

# ----- dispatcher ------------------------------------------------------------
run_module() {
  case "$1" in
    base)   install_base ;;
    pyenv)  install_pyenv ;;
    edge)   install_edge ;;
    argos)  install_argos ;;
    verify) verify_all ;;
    all)    install_base && install_pyenv && install_edge && install_argos && verify_all ;;
    *)      err "Unknown module: $1"; usage; return 1 ;;
  esac
}

usage() { sed -n '2,30p' "$0" | sed 's/^#\s\{0,1\}//'; }

main() {
  [ "$#" -eq 0 ] && { usage; exit 0; }
  for m in "$@"; do run_module "$m" || { err "Module '$m' failed — stopping."; exit 1; }; done
  section "done"
  ok "If you ran 'edge', log out/in before using the NPU."
}

main "$@"
