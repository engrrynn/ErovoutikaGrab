# ErovoutikaGrab: Technical Mass Production Specification
## Software, Hardware, Electrical & Systems Engineering Blueprint

> **Document Classification**: Engineering & Manufacturing Technical Specification  
> **Target Audience**: Hardware Engineers, Embedded Firmware Developers, Systems Architects, Factory SMT/Assembly Teams, and QA Engineers  
> **Companion Ready**: Formatted with quantitative specs, pinout matrices, circuit schematics, and slide-by-slide blueprints optimized for **Gemini Google Slides**

---

## Technical Slide Deck Companion (Quick Overview for Google Slides)

If you are generating a technical briefing or manufacturing review in **Gemini Google Slides**, copy and paste the slides below:

```markdown
---
SLIDE 1: Title Slide
Title: ErovoutikaGrab: Technical Systems & Mass Production Blueprint
Subtitle: Dual-Tier Architecture, Edge AI Vision, Firmware Protocol & Manufacturing SOP
Presenter Notes: Welcome to the ErovoutikaGrab engineering deep-dive. This presentation details the hardware bill of materials, electrical interfaces, firmware state machines, edge AI pipeline, and factory assembly procedures for high-volume manufacturing.

---
SLIDE 2: Architectural Overview (Dual-Tier Compute)
Bullet Points:
• High-Level Controller (HLC): Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4 GHz) running Linux Bookworm.
• Real-Time Controller (RTC): Arduino Nano (ATmega328P @ 16 MHz, 5V logic) handling low-level PWM/servo timing.
• Inter-Processor Link: Isolated RFCOMM Bluetooth Serial (/dev/rfcomm0 @ 9600 baud, 8-N-1) with ASCII packet framing.
• Edge AI Vision: 4-Thread XNNPACK Google LiteRT + PyTorch YOLOE running at 15–30 FPS.
• User Interface: Unified HTTPS Cockpit (Port 5001) + Chromium Wayland View-Only LCD Kiosk.
Presenter Notes: The system cleanly decouples real-time motor/servo safety control (Arduino) from high-level computer vision and web server orchestration (Raspberry Pi 5).

---
SLIDE 3: System Functional Clarification: Categorizer vs. Sorter
Bullet Points:
• Visual Categorizer & Mobile Manipulator: Identifies, measures, and classifies 80 COCO classes and 6 HSV color palettes.
• Not a Sorting Assembly Line: Does not feature mechanical dump bins or high-speed sorting chutes.
• Primary Control Philosophy: Manual arm control is preferred for fine manipulation and safety.
• Automated Grabbing Scenario: Triggers ONLY when 4 strict criteria are satisfied:
  1. Object classified as pickable.
  2. Aligned inside Grasp Sweet Spot (X:324, Y:247 ± 20px).
  3. Calibrated distance verified (30–40 cm).
  4. Target stationary (<20 px/s).
Presenter Notes: Clarifying the functional scope is critical for mass production QA. The robot is an exploratory categorizer and assistive arm, not a stationary sorting machine.

---
SLIDE 4: Bill of Materials (BOM) & Key Components
Bullet Points:
• Single-Board Computer: Raspberry Pi 5 (4GB/8GB LPDDR4X, Broadcom BCM2712).
• Microcontroller: Arduino Nano V3.0 (ATmega328P, CH340 / FTDI USB UART).
• Bluetooth Module: HC-05 Transceiver (EROBOTA75, MAC 20:25:08:00:46:FB).
• Motor Driver: TB6612FNG Dual H-Bridge (1.2A continuous, 3.2A peak per channel).
• Drive Motors: 2x DC Geared Motors (6V–12V, 1:48 reduction, 200 RPM) + 65mm high-traction wheels.
• Servos: 2x MG996R Metal Gear (S1 Shoulder, S2 Elbow) + 1x MG90S Micro Metal Gear (S3 Gripper).
• Camera: USB 2.0 / MIPI CSI-2 Wide-Angle CMOS Sensor (640x480 @ 30 FPS MJPG).
• Display: 5.0" / 7.0" DSI/HDMI Capacitive Touch LCD Panel (800x480).
Presenter Notes: All components have been selected for industrial reliability, multi-vendor availability, and low bill-of-materials cost.

---
SLIDE 5: Power Architecture & Brownout Immunity
Bullet Points:
• Primary Battery: 2S / 3S Li-ion Battery Pack (7.4V / 11.1V, 3500–5000 mAh).
• Dual-Rail Buck Regulation:
  - Logic Rail: 5V / 5.0A dedicated to Raspberry Pi 5 and LCD display.
  - Servo Rail: 5V / 4.0A dedicated exclusively to S1, S2, and S3 servo coils.
• Common Grounding: Star ground topology preventing motor inductive noise from resetting the MCU.
• Staggered Boot Sequence: Firmware enforces 300 ms delays between S1, S2, and S3 power attachment.
• Bulk Decoupling: 470 µF low-ESR electrolytic capacitor across servo power rail.
Presenter Notes: Staggered boot and dual buck converters eliminate the #1 failure mode in mobile robotics: servo stall-induced voltage drops that trigger computer reboots.

---
SLIDE 6: Electrical Pinout & Motor Mapping
Bullet Points:
• Motor Driver (TB6612FNG to Arduino):
  - Channel A (Physical Right): PWMA (D3), AIN1 (D4), AIN2 (D5).
  - Channel B (Physical Left): PWMB (D6), BIN1 (D7), BIN2 (D8).
• Servo Outputs (Arduino Nano):
  - S1 Shoulder: D9 | S2 Elbow: D10 | S3 Gripper: D11.
• Software Transposition: 'swap_left_right: true' in config/robot_config.yaml automatically aligns logical drive commands with physical chassis layout.
Presenter Notes: Standardized pin assignments ensure fast assembly harness routing with zero wiring ambiguity.

---
SLIDE 7: Firmware Architecture & ASCII Protocol
Bullet Points:
• Baud Rate: 9600 baud, 8-N-1 (native HC-05 hardware UART default).
• Packet Framing: <COMMAND:PARAM1,PARAM2,...> with checksum and buffer overflow protection.
• Core Packet Set:
  - <DRIVE:L,R>: Direct Left/Right PWM (-255 to 255).
  - <NUDGE:DIR,MS,PWM>: Precision micro-pulse (anti-stiction burst).
  - <SERVO:S1,S2,S3>: Interpolated multi-servo angular positioning.
  - <STOP>: Instant drive cut and brake.
  - <STATUS:L,R,S1,S2,S3>: 250 ms periodic telemetry heartbeat.
• Hardware Watchdog: Shuts off drive motors if no valid command received for 1,500 ms.
Presenter Notes: The non-blocking millis() scheduler ensures smooth servo interpolation without blocking serial communication or safety watchdogs.

---
SLIDE 8: Kinematic Safety Clamps & Joint Priority
Bullet Points:
• Servo 1 (Shoulder): 60° (Lower Reach) to 170° (Forward Reach) | Neutral: 93°.
• Servo 2 (Elbow): 0° (Floor Extension) to 45° (Mid-Bend) | Neutral: 45°.
• CRITICAL SAFETY CLAMP: S2 is strictly clamped to ≤ 45° in firmware and software to eliminate mechanical linkage binding and servo burn-out.
• Servo 3 (Gripper): 40° (Firm Grip) to 170° (Full Open) | Neutral: 110°.
• Directional Joint Priority:
  - Lowering Arm: S1 extends first, then S2 lowers.
  - Retracting Arm: S2 lifts first away from ground, then S1 stows.
Presenter Notes: The directional joint priority prevents the gripper from dragging across the floor or colliding with the chassis during transitions.

---
SLIDE 9: Motion Dynamics: Smooth Cruise vs. Glide Pulse
Bullet Points:
• Target Velocity Estimator: Computes centroid displacement velocity (V = Δd / Δt).
• Stationary Target (<20 px/s): Activates SMOOTH_FAST cruise mode. Continuous forward drive without frame-to-frame zero-brake stutter.
• Moving Target (>32 px/s): Activates SMOOTH_PULSE mode with slew-rate limiting (±35 PWM/step) easing to glide speed.
• Stiction Jump: Initial transition from standstill (0 PWM) jumps directly to 240 PWM to overcome static tire friction before applying slew smoothing.
• Latency Compensation: Feed-forward velocity compensation compensates for camera + inference latency (93 ms).
Presenter Notes: Slew-rate limiting and velocity hysteresis eliminate the physical jerk and chatter typical of standard visual servoing robots.

---
SLIDE 10: Edge AI Vision & Color Classifier
Bullet Points:
• Primary Vision Models:
  - Google LiteRT: YOLO11 (s/n/m/l), MobileNet SSD v1 (36 ms latency), EfficientDet-Lite0 (47 ms).
  - PyTorch YOLOE: Open-vocabulary open-world object detection and segmentation.
• Sub-Millisecond Color Classifier: Vectorized HSV engine (<0.3 ms latency, ~4,000 FPS throughput) for Red, Green, Blue, Yellow, Orange, Purple.
• Grasp Sweet Spot Calibration: Bounding box centered at X=324, Y=247, Width=196, Height=157.
Presenter Notes: Dual-engine vision ensures sub-millisecond color reaction times while maintaining deep semantic understanding for 80 object categories.

---
SLIDE 11: Network Architecture & Boot Supervisor
Bullet Points:
• Supervisor Service: egrabbot-wifi-init.service runs at system boot.
• Dual Network Modes:
  - AP Hotspot Mode: SSID "Erovoutika_Grab_Bot", WPA2-PSK "egrabbot1234", IP 10.42.0.1.
  - Client Wi-Fi Mode: Seamless connection via NetworkManager 'nmcli con up <ssid>'.
• Key-Mgmt Fix: Explicit '802-11-wireless-security.key-mgmt wpa-psk' ensures flawless connection to modern WPA2/WPA3 access points.
• Boot Preference Persistence: Switching networks at runtime never alters the user's permanent 'startup_mode' in config/network_config.yaml.
Presenter Notes: Network setup is robust and fault-tolerant. If local Wi-Fi is unreachable, the system automatically falls back to AP Hotspot mode.

---
SLIDE 12: Mass Production Flashing & Provisioning Pipeline
Bullet Points:
• Phase 1: PCB & Wire Harness Assembly (JST-XH polarized connectors).
• Phase 2: Arduino Microcontroller Flashing via avrdude (egrabbot.ino.hex).
• Phase 3: Raspberry Pi Golden OS Image Flashing (eMMC / NVMe / MicroSD).
• Phase 4: Automated Hardware Self-Test (scripts/test_bluetooth.py & camera test).
• Phase 5: Fast Pytest Regression Validation (35/35 passing in <20s).
Presenter Notes: The manufacturing SOP is designed for rapid line testing, with complete electrical and functional verification in under 2 minutes per unit.

---
SLIDE 13: End-of-Line (EOL) Quality Assurance Checklist
Bullet Points:
• Visual & Harness Inspection: JST seating, servo horns at neutral (93°, 45°, 110°).
• Voltage Rail Verification: 5.1V logic rail, 5.0V servo rail under 3A load.
• Motor Stall & Trim Verification: Free wheel spin, straight-line drift < 5 cm over 2 meters.
• Optical Sweet-Spot Calibration: Camera focal alignment and sweet-spot crosshair check.
• Bluetooth RFCOMM Handshake: <PING> -> <PONG> verified at 9600 baud.
Presenter Notes: Units pass EOL only when all 10 hardware checkpoints and 35 automated software unit tests pass cleanly.
```

---

## 1. System Engineering Architecture

ErovoutikaGrab employs a dual-tier distributed controller topology. High-latency, computation-intensive tasks (computer vision, web serving, mission sequencing) run on a 64-bit Linux Single-Board Computer (Raspberry Pi 5), while hard real-time, jitter-sensitive tasks (H-bridge motor PWM generation, servo pulse generation, hardware watchdog timing) are offloaded to an 8-bit Microcontroller (Arduino Nano).

```
+-----------------------------------------------------------------------------------+
|                           HIGH-LEVEL CONTROLLER (HLC)                             |
|                           Raspberry Pi 5 (Debian 12)                              |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |   CAMERA SENSOR    |   |     EDGE AI ENGINE    |   |      WEB COCKPIT       |  |
|  |  V4L2 /dev/video0  |-->|  LiteRT (XNNPACK)     |-->|  Flask / HTTPS :5001   |  |
|  |  640x480 @ 30 FPS  |   |  YOLO11 / MobileNet   |   |  MJPEG Video Streamer  |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|                                       │                                           |
|                                       ▼                                           |
|                          +------------------------+                               |
|                          |   VLM MOTION PLANNER   |                               |
|                          |  - Target Dynamics     |                               |
|                          |  - Latency Compensator |                               |
|                          |  - Sweet-Spot Servoing |                               |
|                          +------------------------+                               |
|                                       │                                           |
|                                       ▼                                           |
|                          +------------------------+                               |
|                          | BLUETOOTH SERIAL DRV   |                               |
|                          | /dev/rfcomm0 @ 9600 Bd |                               |
|                          +------------------------+                               |
+---------------------------------------│-------------------------------------------+
                                        │ Bluetooth RFCOMM (UART Frames: <CMD:ARGS>)
                                        ▼
+-----------------------------------------------------------------------------------+
|                           REAL-TIME CONTROLLER (RTC)                              |
|                           Arduino Nano (ATmega328P)                               |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |  PACKET PARSER     |   |   KINEMATIC PLANNER   |   |    SAFETY WATCHDOG     |  |
|  |  ASCII Framed      |-->| Non-Blocking millis() |   | 1500ms Command Timeout |  |
|  |  <DRIVE>, <SERVO>  |   | S1/S2 Priority Step   |   | Auto-Stop Failsafe     |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|            │                          │                                           |
|            ▼                          ▼                                           |
|  +--------------------+   +----------------------------------------------------+  |
|  |  TB6612 H-BRIDGE   |   |                    SERVO COILS                     |  |
|  |  PWMA, PWMB        |   | S1: Shoulder (D9) [60°-170°]                       |  |
|  |  AIN1/2, BIN1/2    |   | S2: Elbow    (D10) [0°-45° CLAMPED]                |  |
|  |  Left/Right Motors |   | S3: Gripper  (D11) [40°-170°]                      |  |
|  +--------------------+   +----------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Complete Bill of Materials (BOM)

| Item # | Subsystem | Component Description | Manufacturer / Part # | Qty | Key Technical Parameters |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | Compute | Single-Board Computer | Raspberry Pi 5 (4GB or 8GB) | 1 | Broadcom BCM2712 Quad Cortex-A76 @ 2.4GHz, LPDDR4X, PCIe 2.0 |
| **2** | Microcontroller | Microcontroller Board | Arduino Nano V3.0 | 1 | ATmega328P @ 16 MHz, 32KB Flash, 2KB SRAM, 5V logic, CH340 USB |
| **3** | Wireless | Bluetooth Transceiver | HC-05 (EROBOTA75) | 1 | Bluetooth 2.0+EDR, SPP Profile, 9600 baud, MAC: `20:25:08:00:46:FB` |
| **4** | Motor Driver | Dual Full-Bridge Driver | TB6612FNG Carrier Board | 1 | Dual MOSFET H-Bridge, 1.2A continuous / 3.2A peak, low $R_{DS(on)}$ |
| **5** | Actuators | DC Geared Drive Motors | TT Gearbox Motor (Dual Shaft) | 2 | 6V–12V DC, 1:48 reduction, ~200 RPM @ 6V, stall torque $\ge 0.8\text{ kg}\cdot\text{cm}$ |
| **6** | Chassis Drive | High-Traction Wheels | 65mm Rubber Tread Wheels | 2 | Molded ABS hub with silicone rubber tire tread, D-shaft coupling |
| **7** | Chassis Support | Omni Support Wheel | 1-inch Ball Caster Wheel | 1 | 360° rotating stainless steel ball caster, nylon housing |
| **8** | Actuators | Arm Shoulder Servo (S1) | TowerPro MG996R Metal Gear | 1 | Stall torque $11.0\text{ kg}\cdot\text{cm}$ @ 4.8V ($13.0\text{ kg}\cdot\text{cm}$ @ 6.0V), copper gears |
| **9** | Actuators | Arm Elbow Servo (S2) | TowerPro MG996R Metal Gear | 1 | Stall torque $11.0\text{ kg}\cdot\text{cm}$, copper gears, calibrated $0^\circ - 45^\circ$ range |
| **10** | Actuators | Gripper Claw Servo (S3)| TowerPro MG90S Micro Metal | 1 | Stall torque $2.2\text{ kg}\cdot\text{cm}$ @ 4.8V, metal gear train |
| **11** | Vision | Wide-Angle Camera Module| USB 2.0 CMOS HD Sensor | 1 | 640x480 @ 30 FPS MJPG, 1/4" sensor, $85^\circ$ diagonal FOV, UVC compliant |
| **12** | Display | Onboard Touch LCD Panel | 5.0" DSI / HDMI Capacitive LCD | 1 | 800x480 resolution, 60Hz refresh, USB/I2C capacitive 5-point touch |
| **13** | Power | Battery Pack | 2S Li-ion 7.4V (or 3S 11.1V) | 1 | 7.4V nominal, 4000–5000 mAh capacity, integrated 15A BMS protection |
| **14** | Power Delivery | Primary Buck Converter | DC-DC Step-Down (Logic Rail) | 1 | Input 6V–24V, Output 5.1V @ 5.0A continuous, ripple < 30 mV |
| **15** | Power Delivery | Secondary Buck Converter| DC-DC Step-Down (Servo Rail) | 1 | Input 6V–24V, Output 5.0V @ 4.0A continuous (dedicated servo power) |
| **16** | Passive Filter | Bulk Decoupling Cap | Aluminum Electrolytic 470 µF | 2 | 470 µF / 25V Low-ESR placed across servo rail and logic rail |
| **17** | Power Control | Master Power Switch | Heavy-Duty Rocker Switch | 1 | SPST 250V/6A rated, chassis snap-in mount |

---

## 3. Electrical Wiring & Interface Matrix

### 3.1 Arduino Nano Pinout Matrix

| Arduino Pin | Function Name | Target Hardware Connection | Signal Type | Electrical Specification |
| :---: | :--- | :--- | :---: | :--- |
| **D0** | RX | HC-05 TXD (Bluetooth Serial) | Input | 3.3V / 5V TTL Serial (9600 baud) |
| **D1** | TX | HC-05 RXD (Bluetooth Serial) | Output | 5V TTL Serial (via 1k/2k divider to 3.3V) |
| **D3** | PWMA | TB6612FNG PWMA (Channel A Speed) | Output (PWM) | 490 Hz PWM, 0–255 duty cycle |
| **D4** | AIN1 | TB6612FNG AIN1 (Channel A Dir 1) | Output (GPIO) | Digital Logic (HIGH/LOW) |
| **D5** | AIN2 | TB6612FNG AIN2 (Channel A Dir 2) | Output (GPIO) | Digital Logic (HIGH/LOW) |
| **D6** | PWMB | TB6612FNG PWMB (Channel B Speed) | Output (PWM) | 980 Hz PWM, 0–255 duty cycle |
| **D7** | BIN1 | TB6612FNG BIN1 (Channel B Dir 1) | Output (GPIO) | Digital Logic (HIGH/LOW) |
| **D8** | BIN2 | TB6612FNG BIN2 (Channel B Dir 2) | Output (GPIO) | Digital Logic (HIGH/LOW) |
| **D9** | SERVO1 | S1 Shoulder Servo Signal Pin | Output (PWM) | 50 Hz Servo Pulse (544–2400 µs) |
| **D10** | SERVO2 | S2 Elbow Servo Signal Pin | Output (PWM) | 50 Hz Servo Pulse (544–2400 µs) |
| **D11** | SERVO3 | S3 Gripper Servo Signal Pin | Output (PWM) | 50 Hz Servo Pulse (544–2400 µs) |
| **5V** | Logic VCC | Logic Power Rail (Buck 1) | Power Input | 5.0V regulated ($\pm 5\%$) |
| **GND** | Ground | Common System Ground | Reference | Star ground connection to battery/bucks |

### 3.2 Motor Driver & Channel Swap Architecture
On the physical ErovoutikaGrab chassis:
* **Channel A** of the H-bridge is wired to the **Physical Right Wheel**.
* **Channel B** of the H-bridge is wired to the **Physical Left Wheel**.

To prevent error-prone physical re-wiring during assembly, the software layer implements a hardware abstraction transpose:
```yaml
# config/robot_config.yaml
motors:
  swap_left_right: true
  trim_offset: 6
  base_speed: 230
  turn_speed: 228
```
When `swap_left_right: true` is set, all logical commands (`send_drive(cmd_l, cmd_r)`) automatically swap channels so that positive forward velocity translates into uniform forward wheel rotation on both sides.

### 3.3 Power Rail Isolation & Brownout Prevention
* **Common Ground**: All grounds (Battery (-), Buck 1 GND, Buck 2 GND, Raspberry Pi GND, Arduino GND, Motor Driver GND, Servo GNDs) must terminate at a single **Star Ground Junction**.
* **Decoupling**: A $470\,\mu\text{F}$ / 25V Low-ESR electrolytic capacitor in parallel with a $0.1\,\mu\text{F}$ ceramic capacitor must be placed directly across the servo power rail pins at the servo distribution bus. This suppresses inductive back-EMF spikes during simultaneous 3-servo stall conditions.
* **Logic Isolation**: The Raspberry Pi 5 and Arduino Nano power rail (Buck 1) is completely isolated from the servo coil power rail (Buck 2). Servo stall currents never siphon current from the SBC logic supply.

---

## 4. Microcontroller Firmware Specification (`arduino_code/egrabbot/egrabbot.ino`)

### 4.1 Communication Protocol: Framed ASCII Packets
The firmware communicates over standard UART at **9600 baud, 8 data bits, no parity, 1 stop bit (8-N-1)**. Commands are enclosed in angle brackets `<...>` to ensure frame synchronization and noise rejection:

| Command Frame | Parameters | Execution Behavior | ACK Response |
| :--- | :--- | :--- | :--- |
| `<PING>` | None | Verifies serial link health | `<PONG>` |
| `<STOP>` | None | Immediately zeroes all motor PWM outputs | `<ACK:STOP>` |
| `<DRIVE:L,R>` | $L, R \in [-255, 255]$ | Sets Left and Right motor H-bridge direction and PWM | None (High throughput) |
| `<NUDGE:DIR,MS,PWM>` | `DIR` $\in \{F,B,L,R\}$, `MS` (ms), `PWM` | Fires a timed drive pulse; auto-stops at timer expiration | `<ACK:NUDGE_DONE>` |
| `<SERVO:S1,S2,S3>` | $S1 \in [60, 170]$, $S2 \in [0, 45]$, $S3 \in [40, 170]$ | Sets target angles; non-blocking millis() updates positions | None |
| `<MACRO:NAME>` | `NAME` $\in \{\text{PICK}, \text{STOW}, \text{CENTER}, \text{DOWN}\}$ | Executes autonomous multi-step articulated kinematic macro | `<ACK:MACRO_DONE>` |
| `<STATUS:...>` | Periodic Telemetry | Emitted every 250 ms: `<STATUS:L,R,S1,S2,S3>` | Telemetry Broadcast |

### 4.2 Kinematic Smoothing & Directional Joint Priority
Direct step jumps in servo angles cause physical overshoot, joint chatter, and excessive peak current draw. The firmware implements a non-blocking `millis()` interpolation loop running at **50 Hz (20 ms interval)**:
* Each tick advances active servos by $\pm 1^\circ$ (for arm) or $\pm 2^\circ$ (for gripper).
* **Direction-Aware Joint Priority**:
  * **When Lowering/Reaching (Arm moving toward ground)**:
    $$\Delta S1 > 0 \implies \text{S1 (Shoulder) extends first; S2 holds posture until S1 reaches target.}$$
  * **When Lifting/Stowing (Arm retracting toward chassis)**:
    $$\Delta S2 > 0 \implies \text{S2 (Elbow) lifts first away from the floor; S1 retracts only after S2 is clear.}$$
  * **Gripper Independence**: Servo 3 (Gripper) moves independently and concurrently so clamping can occur while the arm settles.

### 4.3 Hardware Watchdog Failsafe
To prevent runaway robot conditions caused by Bluetooth dropouts, Wi-Fi crashes, or host application termination:
* The firmware records `lastCommandTime = millis()` upon receiving any valid packet.
* If active drive motors are running ($L \ne 0$ or $R \ne 0$) and no valid packet is received for **$1,500\text{ ms}$**, the firmware automatically executes `stopMotors()` and emits `<WARN:WATCHDOG_STOP>`.

---

## 5. Embedded Linux Software Architecture

### 5.1 Systemd Services
The software stack is managed by three core systemd services configured for auto-restart on failure:

1. **`egrabbot-web.service`**:
   * **ExecStart**: `/usr/bin/python3 /home/egrabbot/ErovoutikaGrab/scripts/web_interface.py --port 5001`
   * **Role**: Primary HTTPS REST API server, MJPEG streaming engine, WebSocket telemetry broadcaster, and hardware abstraction layer. Runs with port 80 automatic HTTP-to-HTTPS redirect.
2. **`egrabbot-wifi-init.service`**:
   * **ExecStart**: `/usr/bin/python3 /home/egrabbot/ErovoutikaGrab/scripts/wifi_manager.py`
   * **Role**: Runs at early boot. Inspects `config/network_config.yaml`. If `startup_mode: wifi` is set, connects to the configured Wi-Fi network; if unreachable or if `startup_mode: ap` is set, initializes the NetworkManager Hotspot (`Erovoutika_Grab_Bot`).
3. **`launch_desktop_gui.py`**:
   * **ExecStart**: Launches Chromium in Wayland kiosk mode pointing to `https://localhost:5001/?device=lcd`.
   * **Role**: Provides the physical onboard touch display with a specialized view-only HUD (touch navigation disabled for students/guests, telemetry gauges and video feed visible).

---

## 6. Edge AI Vision & Motion Control Pipeline

### 6.1 Inference Engine Specification
The vision subsystem runs a multi-tier inference pipeline optimized for ARM64 NEON and 4-threaded XNNPACK acceleration:

| Model File | Framework | Input Resolution | Threads | Inference Latency | Primary Usage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ssd_mobilenet_v1_coco.tflite` | Google LiteRT | 300x300 | 4 | **36 ms** (27.8 FPS) | Ultra-fast mobile tracking & obstacle detection |
| `efficientdet_lite0_coco.tflite` | Google LiteRT | 320x320 | 4 | **47 ms** (21.3 FPS) | Balanced precision person & object tracking |
| `yolo11n.tflite` | Google LiteRT | 320x320 | 4 | **58 ms** (17.2 FPS) | High-accuracy lightweight object sizing |
| `yolo11s.tflite` | Google LiteRT | 320x320 | 4 | **89 ms** (11.2 FPS) | Precision categorization & sweet-spot grasp planning |
| `yoloe-11l-seg-pf.pt` | PyTorch YOLOE | 320x320 | 4 | **180 ms** (5.5 FPS) | Open-vocabulary text prompt segmenter & mask analysis |

### 6.2 Sub-Millisecond HSV Color Classifier
For high-speed color tracking activities, a dedicated vectorized OpenCV HSV classifier processes target frames without deep neural network latency:
* **Execution Time**: **$< 0.3\text{ ms}$** (~4,000 FPS throughput).
* **Color Palettes**: Red, Green, Blue, Yellow, Orange, Purple.
* **Morphology**: $5\times 5$ elliptical structuring element for opening and closing, suppressing lighting speckle and background reflections.

### 6.3 Grasp Sweet-Spot Geometry
The physical camera is mounted rigidly above the chassis pointing forward-downward toward the ground contact zone. The grasping sweet-spot is defined in image pixel coordinates:
$$\text{Center } (X, Y) = (324, 247), \quad \text{Box Width} = 196\text{ px}, \quad \text{Box Height} = 157\text{ px}$$
* **Alignment Tolerances**: $\text{Tolerance}_X = \pm 20\text{ px}$, $\text{Tolerance}_Y = \pm 25\text{ px}$.
* When an object centroid satisfies:
  $$|C_x - 324| \le 20 \quad \text{and} \quad |C_y - 247| \le 25$$
  and standoff distance is verified between $30\text{ cm}$ and $40\text{ cm}$, the target is flagged as `ALIGNED [READY TO GRASP]`.

### 6.4 Target Dynamics & Slew-Rate Motion Limiting
To eliminate mechanical chatter, gearbox wear, and chassis overshoot:
1. **Target Velocity Estimation**:
   $$V_{\text{target}} = \frac{\sqrt{(C_{x, t} - C_{x, t-1})^2 + (C_{y, t} - C_{y, t-1})^2}}{\Delta t} \quad [\text{px/s}]$$
2. **Velocity Hysteresis**:
   * $V_{\text{target}} < 20\text{ px/s} \implies$ **`SMOOTH_FAST` Mode**: Continuous drive forward at target speed without frame pauses.
   * $V_{\text{target}} > 32\text{ px/s} \implies$ **`SMOOTH_PULSE` Mode**: Slew-rate smoothed pulse motion easing down to glide speed rather than slamming hard electric 0-brake locks.
3. **Slew-Rate Limiter**:
   $$\Delta \text{PWM} = \text{clamp}(\text{Target PWM} - \text{Current PWM}, -35, +35)$$
   * Exception (Stiction Break): When starting from a dead stop ($\text{Current PWM} = 0$), the controller immediately commands minimum overcoming speed ($240\text{ PWM}$) on step 1 to break static tire friction before applying slew smoothing.

---

## 7. Calibrated Kinematic Envelopes

```
                                  [SHOULDER S1]
                                     (93° UP)
                                       / \
                                      /   \
                                     /     \
                                    /       \
                                   /         \
                         [ELBOW S2]           \
                          (45° SAFE)           \
                             \                  \
                              \                  \
                               \                  \
                             [GRIPPER S3]          [S1: 170° REACH]
                           (40° CLOSE / 170° OPEN)       \
                                                          \
                                                        [FLOOR LEVEL]
```

| Joint Identifier | Calibrated Min | Calibrated Neutral | Calibrated Max | Mechanical Rationale & Protection |
| :--- | :---: | :---: | :---: | :--- |
| **Servo 1 (Shoulder)** | $60^\circ$ | $93^\circ$ | $170^\circ$ | $93^\circ$ provides balanced upright travel posture; $170^\circ$ extends reach to ground contact. |
| **Servo 2 (Elbow)** | $0^\circ$ | $45^\circ$ | **$45^\circ$ (CLAMP)** | **Strict mechanical safety ceiling**: Angles $>45^\circ$ cause mechanical binding against arm uprights, stripping servo gears. Clamped in firmware and software. |
| **Servo 3 (Gripper)** | $40^\circ$ | $110^\circ$ | $170^\circ$ | $40^\circ$ ensures firm clamp on thin objects without stalling motor coils; $170^\circ$ provides maximum jaw aperture ($55\text{ mm}$). |

---

## 8. Mass Production Assembly & Provisioning SOP

### 8.1 Mechanical & Electrical Assembly Sequence
1. **Chassis Base Assembly**: Mount dual DC gear motors to acrylic chassis using M3x30 screws. Install rubber drive wheels and bolt rear ball caster with M3 standoffs.
2. **Electronics Sub-Assembly**:
   * Mount Arduino Nano and TB6612FNG driver onto the lower carrier PCB.
   * Connect JST-XH wiring harnesses from H-bridge outputs to motor solder lugs.
   * Install dual buck converters and wire common star ground bus.
3. **Robotic Arm Assembly**:
   * Center all servos electrically to $90^\circ$ using a servo tester before attaching horns.
   * Attach S1 shoulder horn at $93^\circ$ (vertical alignment).
   * Attach S2 elbow horn at $45^\circ$. Ensure mechanical travel stops at $45^\circ$ without binding.
   * Attach S3 gripper assembly and verify smooth jaw meshing across $40^\circ - 170^\circ$.
4. **Upper Compute & Display Assembly**:
   * Secure Raspberry Pi 5 onto upper acrylic deck using nylon standoffs.
   * Connect USB camera to USB 3.0 port (lower blue port).
   * Connect LCD display via DSI ribbon cable or HDMI patch lead + USB touch lead.
   * Mount master rocker switch and route main battery leads through switch to buck inputs.

### 8.2 Microcontroller Flashing SOP
Connect Arduino Nano via USB programming cable to the factory provisioning station:
```bash
# Compile and flash Arduino firmware using Arduino CLI
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328old arduino_code/egrabbot/egrabbot.ino
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:nano:cpu=atmega328old arduino_code/egrabbot/egrabbot.ino

# Verify boot banner over serial at 9600 baud
# Expected response: <READY:ErovoutikaGrab_v2.0>
```

### 8.3 Raspberry Pi Golden Image Deployment
1. Write the verified **ErovoutikaGrab Production Golden Image** (`egrabbot-v2.0-golden.img.zst`) to industrial eMMC / MicroSD / NVMe storage using `dd` or Raspberry Pi Imager.
2. The image pre-contains:
   * Debian 12 Bookworm 64-bit with kernel 6.6+
   * Python 3.13 virtual environment with LiteRT, PyTorch, OpenCV, Flask, NetworkManager
   * Configured systemd units: `egrabbot-web.service` and `egrabbot-wifi-init.service`
   * Bluetooth serial auto-bind configuration (`rfcomm bind 0 20:25:08:00:46:FB 1`)

---

## 9. End-of-Line (EOL) Quality Assurance Checklist

Every mass-produced unit must complete and pass this 10-point inspection before factory boxing:

```
[ ] 1. Visual Harness Inspection: All JST-XH connectors seated; no pinched wires under arm linkage.
[ ] 2. Power Rail Verification:
       - Logic Rail measures 5.10V ± 0.10V at Raspberry Pi GPIO Pin 2/4.
       - Servo Rail measures 5.00V ± 0.15V at Servo Distribution Bus under 2.5A simulated load.
[ ] 3. Staggered Boot Verification: Upon power switch ON, verify 300ms audible step delay between S1, S2, S3.
[ ] 4. Safe Joint Calibration Verification:
       - S1 Shoulder moves cleanly between 60° and 170°.
       - S2 Elbow moves cleanly between 0° and 45° with ZERO mechanical binding.
       - S3 Gripper opens to 170° and closes to 40°.
[ ] 5. Motor Drive & Trim Check:
       - Unit drives straight on smooth test track (drift < 5cm over 2 meters).
       - Nudge micro-burst (<NUDGE:F,250,250>) overcomes static friction immediately.
[ ] 6. Bluetooth RFCOMM Handshake:
       - Host sends <PING>; receives <PONG> within 50ms over /dev/rfcomm0.
[ ] 7. Watchdog Failsafe Test:
       - Send <DRIVE:200,200>; sever serial connection; verify motors halt within 1,500ms.
[ ] 8. AI Vision & Camera Test:
       - Video feed streams at 30 FPS over HTTPS /video_feed.
       - Model detects standard test target (bottle) with confidence > 85%.
       - Grasp sweet spot overlay aligns accurately with crosshairs.
[ ] 9. Automated Regression Test Suite:
       - Run: PYTHONPATH=. pytest tests/test_vlm_motion_planner.py tests/test_activities.py
       - Output: 35 passed in < 20 seconds.
[ ] 10. Network Provisioning Check:
       - AP Hotspot broadcasts SSID "Erovoutika_Grab_Bot".
       - Client connects and loads Cockpit GUI at https://10.42.0.1:5001.
```

---

## 10. Technical Contact & Engineering Escalation

* **Hardware & Electrical Lead**: Erovoutika Systems Engineering Team
* **Firmware & Robotics Software**: Robotics Control & Vision Lab
* **Repository Path**: `/home/egrabbot/ErovoutikaGrab`
* **Configuration Manifests**:
  * Robot Kinematics: `config/robot_config.yaml`
  * Vision & Sweet Spot: `config/vision_config.yaml`
  * Network & Modes: `config/network_config.yaml`
