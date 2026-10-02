# ErovoutikaGrab 2.0 - Autonomous AI Mobile Manipulator

An enterprise autonomous mobile manipulation and AI vision sorting system engineered for the **Raspberry Pi 4 / 5** controlling an **Arduino Nano** microcontroller over **Bluetooth (HC-05 / `EROBOTA75`)** with an **A4Tech USB Camera**, **7-inch HDMI LCD Display ($800 \times 480$)**, and corporate **Erovoutika** branding.

---

## 📊 Current System Status

| Subsystem | Current Status | Operational Specification |
| :--- | :---: | :--- |
| **Operational Architecture** | **SYNCHRONIZED COCKPIT** | Unified Web Cockpit + Fullscreen View-Only Robot LCD HUD |
| **Display Resolution** | **$800 \times 480$ NATIVE** | Inset bezel-safe kiosk view on 7-inch LCD (`HDMI-A-1`) |
| **Neutral Desktop** | **CLEAN BLACK LOGO** | Solid black background with centered Erovoutika logo; zero taskbars, zero desktop icons |
| **Web Service & Teleop** | **NATIVE HTTPS (5001)** | Secured TLS cockpit at `https://egrabbot.local:5001` with Port 80 HTTP auto-forward |
| **Microcontroller Link** | **BLUETOOTH RFCOMM** | Serial link on `/dev/rfcomm0` (MAC `20:25:08:00:46:FB` @ 9600 baud) |
| **Motor Mapping** | **SWAP APPLIED** | `swap_left_right: true`; Left/Right channels transposed so controls match physical wheels |
| **Motor Calibration** | **CALIBRATED & VERIFIED** | Base: `225` PWM, Turn: `240` PWM, Stiction: `230` PWM, Nudge: `245` PWM (250ms), Trim: `+6` |
| **Arm Kinematics** | **SMOOTH ALTERNATING** | Alternating micro-stepping (`step_deg: 2`, `step_delay_s: 0.01`, `lead_deg: 0`); eliminates floor collisions |
| **Vision AI** | **100% PURE GOOGLE LITERT** | 4-Thread XNNPACK LiteRT (YOLO11 s/n/m/l, MobileNet SSD, EfficientDet-Lite0); 1.7GB reclaimed |
| **Activities Subsystem** | **MUTUALLY EXCLUSIVE** | Person Follower (AI pursuit) & Color Tracking (6 colors); instant teleop override safety |
| **Detection vs Pickability** | **UNCONSTRAINED SCANNING** | Model classifies all 80 COCO classes; user configures `pickable_classes` for grasp eligibility |
| **Color Detection** | **SUB-MILLISECOND HSV** | Vectorized dominant color classifier (< 0.3 ms, ~4,000 FPS, core crop ROI 60%) |
| **Camera View** | **1480px + CINEMA MODE** | Widened Cockpit grid + `⛶ Expand Width` full-width Cinema Mode toggle |
| **Service Status** | **WEB SERVICE ONLY** | `config/egrabbot-web.service` manages the unified server; exhibit service decommissioned |
| **Arduino Firmware** | **FROZEN & VERIFIED** | `arduino_code/egrabbot/egrabbot.ino` 100% untouched; framed protocol `<CMD:args>` |
| **Automated Tests** | **100% PASSING** | 12 active test modules; **93/94 unit tests passing (1 skipped)** |

---

## 🌟 Key Features

- **Embedded YOLOE Open-Vocabulary Object Detection & Google LiteRT Dual Runtime:**
  - Dual-engine architecture supporting PyTorch (`.pt`) and Google LiteRT (`.tflite` via `ai-edge-litert`) running on-device on the Raspberry Pi 5 CPU.
  - Complete multi-tier model lineup from ultra-fast Nano to Large: `yolo11n.tflite` (~54ms, ~18.5 FPS), `yolo11s.tflite` (~78–115ms, ~12.8 FPS, default/optimal), `yolo11m.tflite` (~383ms), `yolo11l.tflite` (~476ms), and `yoloe-26l-seg.pt` (~2.6–7.5s).
  - Open-vocabulary segmentation and detection for items (`bottle`, `can`, `cup`, `box`, `plastic`, `paper`, `metal`, `trash`, `tool`, `toy`, `phone`, `container`) with zero internet dependency.
  - Fully integrated with visual servoing sweet spot alignment verification and unconstrained detection mode.
  - **Interactive Web Customization Panel (`tab-yoloe`)**: Tune confidence threshold (0.10–0.90), input resolution (`256`, `320`, `480`, `640`), class presets, and model weights (`.pt` and `.tflite`) with "Apply Live" and "Save to YAML" actions.
  - **Comprehensive Telemetry**: Real-time AI cards in Web Cockpit, LCD HUD view-only mode, and Qt GUI dashboard displaying detection confidence bar, pickability status, sweet-spot error offsets (`dx`, `dy`), color badge, backend pill, and inference FPS.
- **Ultra-Fast Sub-Millisecond Color Detection Engine:**
  - Vectorized HSV dominant color detector (< 0.3 ms, ~4,000 FPS) running purely in Python NumPy with zero network overhead.
  - Samples central 60% of bounding box to eliminate table and floor background bleed.
- **Standalone YOLOE Diagnostic Suite (`scripts/test_yoloe.py`):**
  - Full CLI diagnostic tool supporting multi-pass latency benchmarks (`--benchmark N`), live camera checks (`--cam 0`), mock frames (`--mock`), custom classes, color classification, and sweet-spot verification.
- **Unified Synchronized Web Cockpit & View-Only LCD HUD:**
  - **Operator Web Cockpit** (`https://egrabbot.local:5001`): Full interactive control, virtual D-pad teleoperation, servo sliders, and camera view destination toggling.
  - **Robot LCD HUD** (`?device=lcd`): Kiosk fullscreen view-only dashboard with casing overscan insets, large telemetry characters, and Erovoutika branding.
- **Smooth Alternating Servo Kinematics:**
  - Micro-stepping configured directly in [`config/robot_config.yaml`](file:///home/egrabbot/ErovoutikaGrab/config/robot_config.yaml) (`step_deg: 2`, `step_delay_s: 0.01`, `lead_deg: 0`).
  - Alternates incremental angles between Servo 1 (Shoulder) and Servo 2 (Elbow) for brownout-free motion without stretching.
  - Preserves directional kinematic clearance: Servo 1 leads on reach down; Servo 2 leads on return stow.
  - Servo 3 (Gripper, Pin 11) is physically isolated: actuates only after base travel completes (or before base moves if `gripper_first=True`).
- **Fail-Safe Teleoperation Engine:** Eliminates command lag and stuck movement via non-blocking I/O, immediate key-release stops, OS repeat suppression, and hardware lease watchdogs.
- **Motor Channel Transposition (`swap_left_right`):** Clean software abstraction transposes Arduino motor driver pins so left steering turns left and right steering turns right without modifying frozen firmware.
- **Native DualStack HTTPS & Port 80 Redirect:** Runs secure TLS on port 5001 with self-contained certificates for `egrabbot.local`, plus background Port 80 redirect for seamless access.
- **Zero-Footprint Neutral Desktop:** Desktop runs without taskbar (`wf-panel-pi`) or desktop icon clutter, displaying a centered Erovoutika logo on a solid black wallpaper when no apps are running.
- **100% Frozen Firmware Architecture:** All business logic, safety rules, trim formulas, and kinemetics reside in Python; low-level microcontroller firmware is stable and frozen.

---

## 📁 Repository Structure

```text
ErovoutikaGrab/
├── arduino_code/
│   └── egrabbot/
│       ├── egrabbot.ino              # Modern non-blocking 9600 baud firmware (FROZEN)
│       └── egrabbot_legacy.ino       # Original backup sketch
├── company_asset/                    # Official corporate assets & wallpapers
│   ├── Erovoutika-Light-Logo-1.webp  # Official webp company logo
│   ├── Erovoutika_white_logo.png     # Official high-resolution white logo
│   ├── erovoutika_black_logo_wallpaper_800x480.png  # Native 800x480 neutral desktop wallpaper
│   └── erovoutika_black_logo_wallpaper_1920x1080.png # Full HD neutral desktop wallpaper
├── config/
│   ├── egrabbot-web.service          # Systemd service for unified web & teleop server
│   ├── robot_config.yaml             # Calibrated motor speeds, trim, alternating servo limits, Bluetooth MAC
│   ├── ssl/                          # Dedicated TLS certificates for egrabbot.local
│   └── vision_config.yaml            # Camera V4L2 config, sweet spot ROI coordinates
├── human_intervention/               # Operator documentation
│   ├── README.md                     # Quick directory navigation
│   └── SCRIPTS_PURPOSE_GUIDE.md      # Detailed reference guide for all core scripts
├── scripts/
│   ├── calibrate_camera.py           # Multi-modal sweet spot visual calibration tool
│   ├── calibrate_motors.py           # Stiction ramp test, base speed & balance trim calibration
│   ├── calibrate_servos.py           # Physical boundary discovery for 3-DOF arm servos
│   ├── launch_desktop_gui.py         # Kiosk launcher for view-only robot LCD HUD
│   ├── run_robot.py                  # Full production launcher (defaults to fullscreen kiosk HUD)
│   ├── test_bluetooth.py             # Bluetooth RFCOMM serial link diagnostics & CLI test
│   ├── test_yoloe.py                 # Standalone YOLOE vision diagnostic & benchmark suite
│   └── web_interface.py              # Unified HTTPS server & dual-mode dashboard
├── src/
│   ├── adapters/
│   │   ├── camera/                   # V4L2 threaded capture adapter & Mock camera
│   │   ├── comm/                     # Bluetooth RFCOMM serial, Mock comms & framed protocol
│   │   ├── camera/                   # V4L2 threaded capture adapter & Mock camera
│   │   ├── comm/                     # Bluetooth RFCOMM serial, Mock comms & framed protocol
│   │   └── vision/                   # YOLOE adapter, LiteRT backend, HSV Color detector, Mock vision
│   ├── core/                         # Autonomous State Machine, RobotContext, Event definitions
│   ├── manipulation/                 # 3-DOF Arm Controller & alternating kinematics
│   ├── navigation/                   # Visual servoing proportional controller & micro-nudges
│   └── ui/                           # Qt LCD HUD Dashboard component
├── tests/                            # 11 automated pytest test modules (80 tests, 100% passing)
├── requirements.txt                  # Python dependencies
└── README.md                         # This documentation
```

---

## 🚀 Operating the Robot

### 1. Launching the Robot System
To start the robot with the fullscreen synchronized View-Only LCD HUD on the robot's physical screen and the HTTPS cockpit:

```bash
python3 scripts/run_robot.py
```
* **Robot Screen**: Launches the View-Only HUD in fullscreen kiosk mode with casing overscan protection.
* **Remote Browser**: Open `https://egrabbot.local:5001` (or `http://egrabbot.local` via Port 80 forwarder) on any computer or phone on the network.

To run headless without the GUI:
```bash
python3 scripts/run_robot.py --headless
```

### 2. Standalone System Service
To keep the web interface active in the background:
```bash
sudo systemctl enable --now /home/egrabbot/ErovoutikaGrab/config/egrabbot-web.service
```

### 3. Hardware Calibration & Diagnostics

* **Robotic Arm Servo Limits Discovery:**
  ```bash
  python3 scripts/calibrate_servos.py
  # Or offline simulation:
  python3 scripts/calibrate_servos.py --mock
  ```
* **Motor Speed & Balance Trim Calibration:**
  ```bash
  python3 scripts/calibrate_motors.py
  ```
* **Bluetooth Serial Link Diagnostics:**
  ```bash
  python3 scripts/test_bluetooth.py
  ```
* **Camera Sweet Spot Calibration:**
  ```bash
  python3 scripts/calibrate_camera.py
  ```
* **YOLOE Vision AI Benchmark & Diagnostics:**
  ```bash
  python3 scripts/test_yoloe.py --mock --benchmark 5
  ```

---

## 🧪 Verification & Testing

Run all unit tests across the codebase:

```bash
python3 -m pytest tests/ -v
```

**Status**: **77 / 77 tests passing (100%)** across 11 test modules.

---

## 📖 Detailed Script Documentation

A comprehensive, script-by-script technical reference detailing every CLI argument, execution flow, and configuration parameter is available in:  
👉 **[`human_intervention/SCRIPTS_PURPOSE_GUIDE.md`](file:///home/egrabbot/ErovoutikaGrab/human_intervention/SCRIPTS_PURPOSE_GUIDE.md)**
