#!/usr/bin/env bash
# ==============================================================================
# ErovoutikaGrab 2.0 - Mass Production Automated Setup & System Provisioner
# ==============================================================================
# Target Hardware: Raspberry Pi 5 / Pi 4 (aarch64 64-bit)
# Operating System: Raspberry Pi OS (Debian 12 Bookworm / Debian 13 Trixie)
# Microcontroller: Arduino Nano (HC-05 Bluetooth or USB-Serial)
#
# Usage:
#   sudo ./install.sh [--non-interactive] [--skip-apt] [--skip-models]
# ==============================================================================

set -eo pipefail

# Color Codes for Terminal Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Script and Directory Resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME="$(eval echo ~"$TARGET_USER")"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_banner() {
    echo -e "${BOLD}${BLUE}"
    echo "=================================================================="
    echo "   🤖 ErovoutikaGrab 2.0 - Automated Mass Production Installer   "
    echo "   Company: Erovoutika Electronics and Robotics"
    echo "   Target Directory: $SCRIPT_DIR"
    echo "   Target User:      $TARGET_USER ($TARGET_HOME)"
    echo "=================================================================="
    echo -e "${NC}"
}

# ------------------------------------------------------------------------------
# 1. Pre-flight Verifications
# ------------------------------------------------------------------------------
preflight_checks() {
    log_info "Step 1/8: Running pre-flight system checks..."

    # Check root privileges
    if [ "$(id -u)" -ne 0 ]; then
        log_error "This script must be executed with sudo privileges."
        echo "Please re-run: sudo ./install.sh"
        exit 1
    fi

    # Check Architecture
    ARCH="$(uname -m)"
    if [ "$ARCH" != "aarch64" ]; then
        log_warn "Detected CPU architecture '$ARCH'. ErovoutikaGrab LiteRT and YOLOE models are optimized for aarch64 (64-bit OS)."
    else
        log_success "Architecture verified: $ARCH (64-bit ARM)"
    fi

    # Check Available Disk Space (Requires at least 3 GB free)
    FREE_KB="$(df -k "$SCRIPT_DIR" | tail -1 | awk '{print $4}')"
    FREE_GB="$((FREE_KB / 1024 / 1024))"
    if [ "$FREE_GB" -lt 3 ]; then
        log_error "Insufficient disk space: only ${FREE_GB}GB free. At least 3GB free space required."
        exit 1
    fi
    log_success "Disk space verified: ${FREE_GB}GB available."
}

# ------------------------------------------------------------------------------
# 2. System APT Dependencies
# ------------------------------------------------------------------------------
install_system_dependencies() {
    log_info "Step 2/8: Installing APT system packages and hardware drivers..."

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y

    # Core System & Python Development
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        python3-numpy \
        python3-yaml \
        python3-requests \
        python3-serial \
        python3-pytest

    # OpenCV, Media & Video4Linux2 Camera Infrastructure
    apt-get install -y --no-install-recommends \
        python3-opencv \
        v4l-utils \
        libgl1-mesa-glx \
        libglib2.0-0

    # Qt5 Graphical User Interface (for Desktop HUD)
    apt-get install -y --no-install-recommends \
        python3-pyqt5 \
        libqt5gui5 \
        libqt5core5a \
        libqt5widgets5

    # Bluetooth, RFCOMM & Serial Stack
    apt-get install -y --no-install-recommends \
        bluez \
        bluez-tools \
        rfcomm \
        bluetooth

    # NetworkManager (for seamless AP / WiFi management)
    apt-get install -y --no-install-recommends \
        network-manager \
        wireless-tools \
        wpasupplicant

    # Security & Automation Utilities
    apt-get install -y --no-install-recommends \
        openssl \
        curl \
        wget \
        git \
        sudo \
        jq \
        libcap2-bin

    log_success "All system APT packages installed successfully."
}

# ------------------------------------------------------------------------------
# 3. Raspberry Pi Firmware & Hardware Configuration
# ------------------------------------------------------------------------------
configure_firmware() {
    log_info "Step 3/8: Configuring Raspberry Pi firmware and overlays..."

    # Determine firmware config file location (Bookworm uses /boot/firmware/config.txt)
    CONFIG_TXT="/boot/firmware/config.txt"
    if [ ! -f "$CONFIG_TXT" ]; then
        CONFIG_TXT="/boot/config.txt"
    fi

    if [ -f "$CONFIG_TXT" ]; then
        # Create backup if not already created
        if [ ! -f "${CONFIG_TXT}.bak_egrabbot" ]; then
            cp "$CONFIG_TXT" "${CONFIG_TXT}.bak_egrabbot"
            log_info "Created backup of firmware config: ${CONFIG_TXT}.bak_egrabbot"
        fi

        # Ensure camera autodetection and VC4 DRM driver
        declare -A FIRMWARE_OPTS=(
            ["camera_auto_detect"]="1"
            ["display_auto_detect"]="1"
            ["dtoverlay=vc4-kms-v3d"]=""
            ["arm_64bit"]="1"
            ["arm_boost"]="1"
        )

        for OPT in "${!FIRMWARE_OPTS[@]}"; do
            VAL="${FIRMWARE_OPTS[$OPT]}"
            if [ -n "$VAL" ]; then
                if grep -q "^${OPT}=" "$CONFIG_TXT"; then
                    sed -i "s/^${OPT}=.*/${OPT}=${VAL}/" "$CONFIG_TXT"
                else
                    echo "${OPT}=${VAL}" >> "$CONFIG_TXT"
                fi
            else
                if ! grep -q "^${OPT}" "$CONFIG_TXT"; then
                    echo "${OPT}" >> "$CONFIG_TXT"
                fi
            fi
        done
        log_success "Firmware configuration verified in $CONFIG_TXT."
    else
        log_warn "Firmware config.txt not found (non-Raspberry Pi OS). Skipping boot configuration."
    fi
}

# ------------------------------------------------------------------------------
# 4. User Privileges, Sudoers & Group Memberships
# ------------------------------------------------------------------------------
configure_user_permissions() {
    log_info "Step 4/8: Configuring hardware groups and sudo permissions for '$TARGET_USER'..."

    # Add user to required hardware groups
    REQUIRED_GROUPS=("dialout" "video" "netdev" "bluetooth" "render" "input")
    # Add optional RPi groups if they exist
    for GRP in "i2c" "spi" "gpio"; do
        if getent group "$GRP" > /dev/null 2>&1; then
            REQUIRED_GROUPS+=("$GRP")
        fi
    done

    for GRP in "${REQUIRED_GROUPS[@]}"; do
        if getent group "$GRP" > /dev/null 2>&1; then
            usermod -aG "$GRP" "$TARGET_USER"
        fi
    done
    log_success "User '$TARGET_USER' assigned to hardware groups: ${REQUIRED_GROUPS[*]}"

    # Install passwordless sudo rule for rfcomm binding
    SUDOERS_FILE="/etc/sudoers.d/020_egrabbot_rfcomm"
    echo "$TARGET_USER ALL=(ALL) NOPASSWD: /usr/bin/rfcomm" > "$SUDOERS_FILE"
    chmod 0440 "$SUDOERS_FILE"
    visudo -cf "$SUDOERS_FILE" > /dev/null
    log_success "Passwordless rfcomm sudo rule deployed: $SUDOERS_FILE"
}

# ------------------------------------------------------------------------------
# 5. Python Dependency Handling (PEP 668 Compatible)
# ------------------------------------------------------------------------------
install_python_dependencies() {
    log_info "Step 5/8: Installing Python dependencies from requirements.txt..."

    # Ensure pip is up to date
    python3 -m pip install --upgrade pip --break-system-packages || true

    # Install requirements
    REQ_FILE="$SCRIPT_DIR/requirements.txt"
    if [ -f "$REQ_FILE" ]; then
        log_info "Installing packages from $REQ_FILE..."
        python3 -m pip install --break-system-packages -r "$REQ_FILE"
        log_success "Python requirements installed."
    else
        log_warn "requirements.txt not found in $SCRIPT_DIR!"
    fi

    # Verify critical AI inference imports
    log_info "Verifying core Python packages..."
    python3 -c "
import cv2
import serial
import yaml
import requests
import PyQt5
try:
    import ai_edge_litert
    print('[+] LiteRT (ai-edge-litert) successfully verified')
except ImportError:
    print('[-] LiteRT not imported, checking fallback')
try:
    import ultralytics
    print('[+] Ultralytics YOLO successfully verified')
except ImportError:
    print('[-] Ultralytics not imported')
"
}

# ------------------------------------------------------------------------------
# 6. Security & SSL Certificate Provisioning
# ------------------------------------------------------------------------------
configure_ssl_certificates() {
    log_info "Step 6/8: Checking HTTPS / SSL certificate configuration..."

    SSL_DIR="$SCRIPT_DIR/config/ssl"
    mkdir -p "$SSL_DIR"

    CERT_FILE="$SSL_DIR/cert.pem"
    KEY_FILE="$SSL_DIR/key.pem"

    if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
        log_info "Generating new 10-year self-signed SSL certificate for egrabbot.local..."
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$KEY_FILE" \
            -out "$CERT_FILE" \
            -days 3650 \
            -subj "/C=PH/ST=NCR/L=Manila/O=Erovoutika/OU=Robotics/CN=egrabbot.local" \
            -addext "subjectAltName = DNS:egrabbot.local,DNS:localhost,IP:127.0.0.1,IP:192.168.254.130,IP:10.42.0.1" \
            2>/dev/null
        chown -R "$TARGET_USER:$TARGET_USER" "$SSL_DIR"
        chmod 600 "$KEY_FILE"
        chmod 644 "$CERT_FILE"
        log_success "Generated self-contained SSL certificate: $CERT_FILE"
    else
        log_success "Existing SSL certificate found: $CERT_FILE"
    fi
}

# ------------------------------------------------------------------------------
# 7. Systemd Services & Desktop HUD Installation
# ------------------------------------------------------------------------------
install_systemd_services() {
    log_info "Step 7/8: Deploying and enabling systemd background services..."

    # 1. egrabbot-web.service
    WEB_SVC="/etc/systemd/system/egrabbot-web.service"
    cat <<EOF > "$WEB_SVC"
[Unit]
Description=ErovoutikaGrab Web Teleoperation & Calibration Dashboard
After=network.target bluetooth.target rfcomm-bind.service
Wants=rfcomm-bind.service

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/scripts/web_interface.py --port 5001
Restart=always
RestartSec=3
KillMode=mixed
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 "$WEB_SVC"
    log_success "Configured: $WEB_SVC"

    # 2. egrabbot-wifi-init.service
    WIFI_SVC="/etc/systemd/system/egrabbot-wifi-init.service"
    cat <<EOF > "$WIFI_SVC"
[Unit]
Description=E-GrabBot Wireless & Hotspot Mode Startup Initializer
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
User=root
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/scripts/wifi_manager.py init
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 "$WIFI_SVC"
    log_success "Configured: $WIFI_SVC"

    # 3. rfcomm-bind.service
    RFCOMM_SVC="/etc/systemd/system/rfcomm-bind.service"
    # Extract Bluetooth MAC from robot_config if available
    BT_MAC="20:25:08:00:46:FB"
    if [ -f "$SCRIPT_DIR/config/robot_config.yaml" ]; then
        CONF_MAC="$(grep -E "^\s*mac_address:" "$SCRIPT_DIR/config/robot_config.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")"
        if [ -n "$CONF_MAC" ]; then
            BT_MAC="$CONF_MAC"
        fi
    fi

    cat <<EOF > "$RFCOMM_SVC"
[Unit]
Description=Bind Bluetooth RFCOMM0 for ErovoutikaGrab Arduino
After=bluetooth.service
Requires=bluetooth.service
Before=egrabbot-web.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/rfcomm bind 0 $BT_MAC 1
ExecStop=/usr/bin/rfcomm release 0

[Install]
WantedBy=multi-user.target
EOF
    chmod 644 "$RFCOMM_SVC"
    log_success "Configured: $RFCOMM_SVC (Arduino HC-05 MAC: $BT_MAC)"

    # Reload systemd and enable services
    systemctl daemon-reload
    systemctl enable egrabbot-web.service egrabbot-wifi-init.service rfcomm-bind.service
    log_success "Enabled services: egrabbot-web.service, egrabbot-wifi-init.service, rfcomm-bind.service"

    # 4. Deploy Desktop HUD Shortcut
    DESKTOP_DIR="$TARGET_HOME/Desktop"
    if [ -d "$DESKTOP_DIR" ]; then
        HUD_SRC="$SCRIPT_DIR/config/ErovoutikaGrab_HUD.desktop"
        HUD_DEST="$DESKTOP_DIR/ErovoutikaGrab_HUD.desktop"
        if [ -f "$HUD_SRC" ]; then
            sed -e "s|/home/egrabbot/ErovoutikaGrab|$SCRIPT_DIR|g" "$HUD_SRC" > "$HUD_DEST"
            chown "$TARGET_USER:$TARGET_USER" "$HUD_DEST"
            chmod +x "$HUD_DEST"
            log_success "Desktop HUD shortcut deployed to: $HUD_DEST"
        fi
    fi

    # Set proper ownership across repository files
    chown -R "$TARGET_USER:$TARGET_USER" "$SCRIPT_DIR"
}

# ------------------------------------------------------------------------------
# 8. Post-Installation Verification & Health Check
# ------------------------------------------------------------------------------
post_install_verification() {
    log_info "Step 8/8: Performing post-installation health check..."

    # Ensure scripts have execute permissions
    chmod +x "$SCRIPT_DIR/scripts/"*.py || true
    chmod +x "$SCRIPT_DIR/install.sh" || true

    # Check V4L2 cameras
    if command -v v4l2-ctl > /dev/null 2>&1; then
        CAM_COUNT="$(v4l2-ctl --list-devices 2>/dev/null | grep -c "/dev/video" || true)"
        if [ "$CAM_COUNT" -gt 0 ]; then
            log_success "Camera device detected ($CAM_COUNT video nodes found)."
        else
            log_warn "No USB or CSI camera currently detected. Connect A4Tech camera before starting teleop."
        fi
    fi

    # Check Bluetooth controller
    if command -v bluetoothctl > /dev/null 2>&1; then
        if bluetoothctl show | grep -q "Powered: yes"; then
            log_success "Bluetooth controller is powered and ready."
        else
            log_warn "Bluetooth controller is inactive or powered off."
        fi
    fi

    # Run quick unit test suite
    log_info "Executing unit test suite..."
    su - "$TARGET_USER" -c "cd '$SCRIPT_DIR' && PYTHONPATH=. pytest tests/test_vlm_motion_planner.py tests/test_activities.py -q" || {
        log_warn "Unit tests returned non-zero. Please review test output."
    }

    # Start or restart egrabbot-web service
    log_info "Restarting egrabbot-web.service..."
    systemctl restart egrabbot-web.service

    echo ""
    echo -e "${GREEN}${BOLD}=================================================================="
    echo "   🎉 ErovoutikaGrab 2.0 Mass Production Setup Completed!        "
    echo "=================================================================="
    echo -e "${NC}"
    echo -e "Web Cockpit (HTTPS):  ${CYAN}https://egrabbot.local:5001${NC} or ${CYAN}https://<IP>:5001${NC}"
    echo -e "Robot LCD HUD:        ${CYAN}https://localhost:5001/?device=lcd${NC}"
    echo -e "Service Status:       ${YELLOW}sudo systemctl status egrabbot-web.service${NC}"
    echo -e "Live Logs:            ${YELLOW}journalctl -u egrabbot-web.service -f${NC}"
    echo ""
    echo -e "${BOLD}Next Steps for Mass Production:${NC}"
    echo "  1. If pairing a new Arduino HC-05 module, set MAC in: config/robot_config.yaml"
    echo "  2. Reboot once to apply any new firmware video/camera device overlays: sudo reboot"
    echo "=================================================================="
}

# Main Execution Flow
preflight_checks
install_system_dependencies
configure_firmware
configure_user_permissions
install_python_dependencies
configure_ssl_certificates
install_systemd_services
post_install_verification
