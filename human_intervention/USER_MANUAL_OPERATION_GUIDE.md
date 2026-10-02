# ErovoutikaGrab: Non-Technical User Manual & Operating Guide
> **Document Purpose**: Complete, plain-language operating manual and presentation script for end-users, operators, students, and educators.  
> **Target Audience**: Non-technical users, school instructors, robotics demonstrators, lab assistants, and event presenters.  
> **Companion Ready**: Structured with distinct, modular sections and slide-by-slide blueprints optimized to feed directly into **Gemini Google Slides**.

---

## Slide Deck Companion (Quick Overview for Google Slides)

If you are creating a Google Slides presentation with **Gemini in Google Slides**, copy and paste the slides below:

```markdown
---
SLIDE 1: Title Slide
Title: ErovoutikaGrab Smart Robotic Manipulator
Subtitle: Intelligent Object Categorization, Visual Tracking & Mobile Manipulation
Presenter Notes: Welcome to the ErovoutikaGrab system overview. Today we explore an intelligent mobile robot equipped with AI computer vision, dual differential drive, and a 3-axis robotic gripper arm.

---
SLIDE 2: What is ErovoutikaGrab? (Categorizer vs. Sorter)
Bullet Points:
• Visual Categorizer & Tracker: Uses onboard AI camera to identify, size, and classify objects.
• Not an Automated Sorting Line: It does not mechanically dump or sort items into conveyor bins.
• Assistive Mobile Manipulator: Follows targets, inspects items, and approaches objects.
• Hybrid Arm Control: Manual arm operation is preferred for precision, with optional automated grabbing.
Presenter Notes: It is essential to understand that ErovoutikaGrab is an intelligent categorizer and explorer rather than a high-speed factory sorter. It combines autonomous perception with human-directed manipulation.

---
SLIDE 3: Robot Anatomy & Hardware Highlights
Bullet Points:
• Mobile Base: Dual high-traction rubber wheels + rear caster for tight zero-radius turns.
• 3-DoF Robotic Arm: Shoulder (S1), Elbow (S2 with safety limit), and Gripper (S3).
• AI Vision Head: High-definition wide-angle camera for real-time tracking and classification.
• Onboard Touchscreen: Live HUD dashboard showing battery, angles, and camera feed.
• Dual Connectivity: Works anywhere via standalone Hotspot (AP) or local Wi-Fi.
Presenter Notes: Everything is self-contained. The robot houses its own computer, battery system, and LCD screen, and communicates with your phone or laptop wirelessly.

---
SLIDE 4: Powering Up & The Boot Sequence
Bullet Points:
• Master Power Switch: Located on the main chassis rear.
• Safe Staggered Boot: Servos wake up sequentially (S1 -> S2 -> S3) to protect battery voltage.
• Auto-Centering: Arm moves to its safe travel posture (Shoulder 93°, Elbow 45°, Gripper Neutral).
• Network Ready: Robot broadcasts its Wi-Fi hotspot or connects to your saved office network.
Presenter Notes: When powering on, you will hear the servos gently click into position one by one. This staggered boot prevents battery brownouts and protects internal gears.

---
SLIDE 5: Connecting to the Robot Cockpit
Bullet Points:
• Direct Hotspot Mode: Connect phone/PC to "Erovoutika_Grab_Bot" (Password: egrabbot1234).
• Local Wi-Fi Mode: Connect both robot and laptop to the same network (e.g., GFiber_ef820).
• Open Any Web Browser: Navigate to https://egrabbot.local:5001 or http://10.42.0.1:5001.
• Zero Software Install: Complete control via modern web browser on iPhone, Android, Mac, or PC.
Presenter Notes: You do not need to install an app from the App Store. Any web browser provides the full cockpit with live video and controls.

---
SLIDE 6: Cockpit Dashboard Tour
Bullet Points:
• Live Video Window: Real-time camera stream with green AI bounding boxes and distance readouts.
• Virtual D-Pad: Forward, Backward, Left Turn, Right Turn, and Emergency Stop.
• Precision Nudge: Tap nudge buttons for micro-adjustments when lining up objects.
• Speed Sliders: Adjust driving speed from slow educational crawls to brisk navigation.
• Arm Control Sliders: Directly move the Shoulder, Elbow, and Gripper in real time.
Presenter Notes: The cockpit is divided cleanly into visual feedback on the left and responsive tactile controls on the right.

---
SLIDE 7: Arm Operation: Why Manual Control is Preferred
Bullet Points:
• Deliberate & Safe: Manual sliders provide direct human judgment and tactile feedback.
• Protection Against Collisions: Prevents accidental arm strikes against obstacles or fragile objects.
• Preset One-Tap Poses:
  - Center: Balances the arm in neutral upright posture.
  - Stow: Folds the arm safely over the chassis for brisk driving.
  - Reach Down: Lowers the gripper toward the floor.
  - Open / Close Gripper: Gentle, secure clamping.
Presenter Notes: Real-world objects vary widely in shape and fragility. Manual control allows the operator to execute delicate grasps with confidence.

---
SLIDE 8: When Does Automated Grabbing Happen?
Bullet Points:
• Automated Grabbing Scenario: Activates ONLY when specific visual conditions are met:
  1. Target Categorized: Model detects and confirms an eligible object (e.g., bottle, cup).
  2. Sweet-Spot Alignment: Object is centered within the target crosshairs (X:324, Y:247).
  3. Verified Standoff Distance: Robot approaches until the object is 30–40 cm away.
  4. Target Stationary: Object is still (<20 px/s) to ensure safe mechanical contact.
• One-Tap Automated Pick: Press "Execute VLM Grasp" to perform the calibrated grab sequence.
Presenter Notes: Automated grabbing is not random—it requires strict visual confirmation, precise centering, and stationary targets.

---
SLIDE 9: Autonomous Activity: Human Following
Bullet Points:
• AI Person Tracking: Locks onto a person standing in front of the camera.
• Smooth Motion Dynamics:
  - Cruises smoothly when the person stands still or walks steadily.
  - Automatically slows down and pulses when the person makes sudden moves.
• Safety Standoff: Automatically stops 45 cm away to avoid stepping on feet or bumping knees.
• Stationary Mode Toggle: Keeps the body still while panning the camera and tracking with the arm.
Presenter Notes: In human following, the robot behaves like an attentive assistant, maintaining a courteous following distance.

---
SLIDE 10: Autonomous Activity: Object Tracking & Sizing
Bullet Points:
• Object Tracking: Tracks bottles, cups, and balls; steers the chassis to keep them centered.
• Object Sizing: Analyzes object bounding boxes to estimate physical width, height, and distance.
• Live HUD Telemetry: Displays real-time millimeter dimensions and distance on the LCD.
• Categorization Output: Reports object class and confidence score instantly.
Presenter Notes: Ideal for STEM demonstrations: students can place different objects in front of the robot and observe real-time dimensional analysis.

---
SLIDE 11: Autonomous Activity: Color Tracking & Classification
Bullet Points:
• Sub-Millisecond Color Engine: Instantly detects Red, Green, Blue, Yellow, Orange, and Purple.
• Combined Tracking & Classification: Identifies both the color and object identity simultaneously (e.g., "Green Bottle").
• Interactive Chase: Move a colored card or ball, and the robot turns and pursues it smoothly.
Presenter Notes: This activity showcases high-speed perception, processing color spaces at over 3,000 frames per second.

---
SLIDE 12: Sequence Queueing & Exhibit Showcase
Bullet Points:
• Movement Sequence Queueing: Chain custom steps (e.g., Forward 2s -> Turn Right -> Lower Arm).
• Voice Control Ready: Issue spoken commands from your phone browser ("Stop", "Forward", "Grab").
• Exhibit / Showcase Mode: Automated self-demonstration sequence for science fairs and exhibitions.
Presenter Notes: Exhibit mode allows the robot to introduce itself and cycle through its arm articulation autonomously.

---
SLIDE 13: Safety, Battery & Troubleshooting
Bullet Points:
• Red Emergency Stop: Instantly halts all motor drives and freezes the robotic arm.
• Battery Health: Monitor voltage gauge on HUD; recharge when battery drops below 20%.
• Mechanical Binding Safeguard: Elbow joint is permanently limited to 45° to protect gears.
• Auto-Watchdog: Robot stops automatically within 1.5 seconds if Wi-Fi or Bluetooth disconnects.
Presenter Notes: Safety is built in at every level, from firmware watchdogs to physical joint limits.
```

---

## 1. System Overview: What is ErovoutikaGrab?

The **ErovoutikaGrab** is an autonomous mobile robot equipped with a high-definition computer vision camera, a two-wheel differential drive mobile chassis, and an articulated 3-Axis robotic arm with a mechanical gripper.

### What ErovoutikaGrab IS:
* **An Intelligent Object Categorizer**: The robot uses advanced artificial intelligence (AI) to look through its camera, detect objects, identify their categories (e.g., bottles, cups, apples, backpacks), measure their apparent sizes, and detect their colors.
* **A Visual Tracking Assistant**: It can autonomously follow a person, follow an object, or chase a specific color.
* **A Mobile Manipulator**: It can drive across flat indoor surfaces to reach objects, position itself in front of them, and pick them up.

### What ErovoutikaGrab IS NOT:
* **It is NOT an Industrial Trash Sorter**: The robot is not designed to sit on a conveyor belt and rapidly throw items into recycling bins. It is a research, educational, and assistive platform designed for targeted categorization, visual servoing, and deliberate manipulation.
* **It is NOT a Blind Bulldozer**: The robot moves only when its camera and AI model have perceived what is in front of it. It continuously checks distance and velocity before driving forward.

---

## 2. Hardware Anatomy & Component Guide

| Component | Location | Description & Everyday Function |
| :--- | :--- | :--- |
| **Mobile Chassis** | Lower Base | Acrylic/metal platform carrying dual high-traction drive wheels and a rear 360° caster wheel. Allows the robot to spin on the spot (zero-radius turn). |
| **Drive Motors** | Base Underside | Dual DC geared motors powered by an internal H-bridge driver. Controlled via speed sliders and nudge buttons. |
| **Robotic Shoulder (S1)** | Lower Arm Joint | Controls arm elevation ($60^\circ$ to $170^\circ$). Moves forward to reach objects and upright ($93^\circ$) for travel. |
| **Robotic Elbow (S2)** | Mid-Arm Joint | Articulates the forearm ($0^\circ$ to $45^\circ$). **Permanently clamped to a maximum of $45^\circ$** to eliminate mechanical binding and gear strain. |
| **Gripper Jaws (S3)** | Arm End-Effector | Mechanical 2-finger claw ($40^\circ$ closed to $170^\circ$ wide open) that clamps around cylindrical or box-shaped items. |
| **AI Camera** | Front Center | 640x480 wide-angle color camera providing continuous video, object detection, and color classification. |
| **Touchscreen LCD** | Top Rear | Displays the real-time View-Only HUD: battery status, joint angles, current activity, and live video stream. |
| **Master Power Switch** | Rear Panel | Rocker switch that turns the entire system on or off. |
| **Rechargeable Battery** | Under-chassis Bay | Powers both the onboard computer and the high-torque servo motors. |

---

## 3. Power-Up Sequence & First-Time Setup

### Step 1: Physical Placement
1. Place ErovoutikaGrab on a flat, clean floor or wide table surface.
2. Ensure there are no loose cables or obstructions within 1 meter of the robot.
3. Make sure the robotic arm is free to move and not touching any walls or packing foam.

### Step 2: Powering On
1. Flip the master rocker switch on the rear panel to **ON**.
2. **Observe the Staggered Boot Sequence**:
   * The onboard computer boots up (takes approximately 25–35 seconds).
   * The servos will wake up sequentially with a 300 ms delay between each joint (Shoulder -> Elbow -> Gripper). This is normal and intentional—it prevents battery voltage dips (brownouts).
   * The arm will automatically position itself in its safe **STOW** travel posture:
     - Shoulder (S1): **93°** (Neutral Upright)
     - Elbow (S2): **45°** (Safe Center)
     - Gripper (S3): **110°** (Neutral Rest)
   * The robot's LCD screen will light up and display the desktop HUD.

### Step 3: Wi-Fi Connection
ErovoutikaGrab can connect in two different ways:

#### Option A: Standalone Hotspot Mode (Works Anywhere)
* If no external Wi-Fi router is available, the robot broadcasts its own private Wi-Fi network:
  * **Wi-Fi Network (SSID)**: `Erovoutika_Grab_Bot`
  * **Password**: `egrabbot1234`
* Connect your smartphone, tablet, or laptop to `Erovoutika_Grab_Bot`.

#### Option B: Local Office / Home Wi-Fi Mode
* If the robot has already been paired with your local network (e.g., `GFiber_ef820`), it connects automatically on boot.
* Connect your phone or laptop to the **same** local Wi-Fi router.

### Step 4: Opening the Web Cockpit
1. Open any modern web browser (Google Chrome, Apple Safari, Microsoft Edge, or Mozilla Firefox).
2. Enter one of the following addresses:
   * Hotspot Mode: `https://10.42.0.1:5001` or `http://10.42.0.1`
   * Local Wi-Fi Mode: `https://egrabbot.local:5001` or `https://<ROBOT-IP>:5001` *(the robot's IP is shown on the top bar of the robot's LCD screen)*.
3. *Security Notice*: If your browser displays a "Connection is not private" warning, click **Advanced** -> **Proceed to egrabbot.local (unsafe)**. This happens because the robot uses a self-contained local security certificate.

---

## 4. Cockpit User Interface Guide

```
+---------------------------------------------------------------------------------------+
|  EROVOUTIKA GRAB COCKPIT                     [BATTERY: 78%] [STATUS: CONNECTED] [STOP]|
+-------------------------------------------+-------------------------------------------+
|                                           |  MANUAL DRIVE CONTROLS                    |
|   LIVE AI CAMERA STREAM                   |                                           |
|   +-----------------------------------+   |         [  FORWARD  ]                     |
|   |  Target: Bottle (94%)             |   |  [ LEFT ] [ STOP  ] [ RIGHT ]             |
|   |                                   |   |         [ BACKWARD  ]                     |
|   |          [+] Crosshair            |   |                                           |
|   |       +-----------------+         |   |  MICRO-NUDGES: [^] [v] [<] [>]            |
|   |       | GRASP SWEET-SPOT|         |   |  SPEED: [=====o==========] 225 PWM        |
|   |       +-----------------+         |   +-------------------------------------------+
|   |                                   |   |  ROBOTIC ARM CONTROLS                     |
|   +-----------------------------------+   |   Shoulder (S1): [====o=====] 93°         |
|   Sweet Spot Status: ALIGNED [READY]      |   Elbow (S2):    [=======o==] 45°         |
|   Distance: 34 cm | Offset: DX: 4, DY: -2 |   Gripper (S3):  [=o========] 170° (Open) |
|                                           |                                           |
|   CURRENT ACTIVITY: Object Tracking       |   PRESET POSES:                           |
|   [ MOTION: ENABLED (TOGGLE) ]            |   [CENTER] [STOW] [REACH DOWN] [GRAB]     |
+-------------------------------------------+-------------------------------------------+
```

### Key Elements of the Screen:
1. **Live Camera Feed**: Shows exactly what the robot sees. When an object is identified, a colored bounding box appears around it with its name and confidence percentage.
2. **Grasp Sweet-Spot Box**: A yellow/green rectangular guide in the center of the video. When a target object enters this box, the robot is perfectly aligned to reach and grasp it.
3. **Drive Controls**:
   * **D-Pad**: Press and hold to drive forward, backward, or rotate.
   * **Nudge Buttons**: Tap once for an ultra-short (250 ms) micro-burst of movement. Perfect for small lining-up adjustments without overshooting.
4. **Arm Sliders & Presets**:
   * Sliders allow gradual, millidegree-level control of each joint.
   * Quick buttons let you instantly stow the arm or prepare the claw.
5. **Red STOP Button**: Instantly cuts motor power and pauses activities in an emergency.

---

## 5. Controlling the Robotic Arm: Manual Preference vs. Automated Grabbing

### Why Manual Arm Control is Preferred
In real-world environments, objects come in different materials, weights, and fragile shapes (thin plastic cups, metal cans, glass bottles, pens).
* **Operator Precision**: Manual control via the sliders or preset buttons allows the operator to judge the exact height, grip width, and approach angle needed.
* **Joint Safety**: Manual control prevents collisions with tables, floors, or side obstacles.
* **Predictability**: The operator maintains total command over when the gripper closes and when the arm lifts.

### The Standard Manual Grasp Workflow
1. Use the **D-Pad** or **Nudges** to drive the robot until the object is directly in front of the wheels (30–35 cm away).
2. Tap **[OPEN GRIPPER]** (`S3 -> 170°`).
3. Tap **[REACH DOWN]** (`S1 -> 170°, S2 -> 0°`). The arm extends forward and lowers the claw to floor level.
4. Tap **[CLOSE GRIPPER]** or drag the Gripper slider to `70°–100°` until the jaws contact the object firmly.
5. Tap **[STOW]** or **[CENTER]**. The elbow automatically lifts first, lifting the object smoothly off the ground, followed by the shoulder retracting to travel posture.

### When Does Automated Grabbing Occur?
ErovoutikaGrab features an intelligent **Vision-Language-Motion (VLM)** planner that *can* automate the grab, but it strictly requires specific physical conditions:

```
[TARGET DETECTED] 
       │
       ▼
[IS OBJECT IN GRASP SWEET SPOT?] ──► NO ──► Drive chassis to align (dx, dy < 20px)
       │ YES
       ▼
[IS DISTANCE 30 - 40 CM?] ─────────► NO ──► Drive chassis forward/backward
       │ YES
       ▼
[IS OBJECT STATIONARY? (<20 px/s)] ─► NO ──► Wait for object to stop moving
       │ YES
       ▼
[CRITERIA SATISFIED] ─────────────► Press "Execute VLM Grasp" or run Auto Grab:
                                     1. Gripper opens to 170°
                                     2. Shoulder & Elbow lower to calibrated floor contact
                                     3. Gripper clamps firmly
                                     4. Arm lifts and returns to Stow position
```

> **Important**: If the object is moving, outside the sweet spot, or too far away, the robot will **not** attempt an automated grab. This prevents missed grabs, knocked-over objects, and mechanical strain.

---

## 6. Guide to Autonomous Activities

The robot includes six primary autonomous activities, plus three interactive modes. Each can be started and stopped with one tap from the **Activities** panel.

### Activity 1: Human Following (`person_follower`)
* **What It Does**: Uses AI person detection to recognize a human standing or walking in front of the robot.
* **How It Moves**:
  * **Stationary Person**: The robot cruises toward the person quickly and smoothly without stuttering.
  * **Moving Person**: The robot uses smooth pulse motion, gently pacing its approach without jerky electric braking.
  * **Safety Margin**: The robot automatically stops at **45 cm** distance. It will never drive directly into your legs.
* **Stationary Mode Toggle**: If you turn off chassis motion, the robot stands in place, swiveling its wheels slightly and tilting its arm to keep the person centered in the camera.

### Activity 2: Object Tracking (`object_tracking`)
* **What It Does**: Locks onto a specific target object class (e.g., `bottle`, `cup`, `sports ball`, `backpack`).
* **How It Moves**: Steers the wheels to keep the target in the center of the camera and maintains a 40 cm following distance.
* **Role**: Visual pursuit and guidance.

### Activity 3: Color Tracking (`color_tracking`)
* **What It Does**: Tracks saturated colors (Red, Green, Blue, Yellow, Orange, Purple) using a vectorized HSV color detector.
* **How It Moves**: Pursues colored objects or cards held in front of the lens.
* **Role**: High-speed, responsive visual servoing demonstration.

### Activity 4: Color Track & Object Classification (`color_track_and_classify`)
* **What It Does**: Combines object classification with color detection. For example, if a green bottle is shown, it identifies both that it is a *bottle* and that its dominant color is *Green*.
* **Role**: Multimodal categorization.

### Activity 5: Object Sizing (`object_sizing`)
* **What It Does**: Computes the pixel dimensions of the object's bounding box and calculates its estimated real-world width, height, and standoff distance.
* **Display**: Shows real-time sizing measurements on the robot's LCD HUD.
* **Role**: Inspection, quality control demonstration, and STEM measurement labs.

### Activity 6: Simple Obstacle Avoidance (`obstacle_avoidance`)
* **What It Does**: Uses camera-based spatial grid analysis to detect when large obstacles or walls fill the camera view.
* **How It Moves**: Drives forward at cruise speed (210 PWM). When an obstacle is detected within its threshold, it smoothly pivots away (turn speed 230 PWM) to clear the path.

### Activity 7: Movement Sequence Queueing
* **What It Does**: Allows the operator to build custom multi-step routines.
* **Example Sequence**:
  1. `Forward` for 2.0 seconds.
  2. `Turn Left` for 0.8 seconds.
  3. `Reach Down` arm pose.
  4. `Close Gripper`.
  5. `Stow Arm`.
* **Role**: Autonomous maze solving, choreographies, and classroom programming exercises.

### Activity 8: Voice Control via Phone
* **What It Does**: Allows the operator to speak into their smartphone browser microphone to trigger commands:
  * *"Forward"*, *"Backward"*, *"Turn Left"*, *"Turn Right"*, *"Stop"*.
  * *"Open Gripper"*, *"Close Gripper"*, *"Reach Down"*, *"Stow Arm"*.
* **Role**: Hands-free accessibility and impressive interactive presentations.

### Activity 9: Exhibit / Showcase Mode
* **What It Does**: A built-in self-demonstration routine.
* **Action**: When triggered, the robot introduces its mechanical range:
  1. Centers the arm upright.
  2. Opens the gripper wide.
  3. Lowers the arm into pick posture.
  4. Clamps the gripper firmly.
  5. Lifts and stows the arm smoothly.
  6. Releases the gripper and returns to neutral.
* **Role**: Perfect for hands-off booth displays, science fairs, and school open houses.

---

## 7. Understanding Robot Motion Dynamics

### Smooth Fast Cruise vs. Smooth Pulse
Users often ask: *Why does the robot sometimes drive continuously and other times move in pulses?*

1. **When the Target is Stationary (Smooth Fast Cruise)**:
   * When an object or person is still, the robot knows its position is stable.
   * It drives forward smoothly and briskly at full speed without stopping between frames.
2. **When the Target is Moving (Smooth Glide Pulse)**:
   * When a person walks or an object is handed back and forth, its position shifts rapidly.
   * To prevent overshooting or bumping into the object, the robot applies smooth, controlled pulses.
   * Unlike older jerky robots, ErovoutikaGrab uses **slew-rate gliding** ($\pm 35$ PWM per step), easing down smoothly without mechanical chatter or gearbox shudder.
3. **Stationary Mode (Body Still, Arm Tracks)**:
   * When you disable driving motion in the activities panel, the robot keeps its wheels planted.
   * It uses its shoulder and elbow servos like an inverse kinematics pan/tilt mechanism to follow the object up and down with the camera.

---

## 8. User Settings & Customization Guide

All settings can be customized directly through the Cockpit Web Interface without writing code:

### Motor Settings
* **Base Speed**: Sets regular forward/backward driving speed (Default: `225–230 PWM`).
* **Turn Speed**: Sets turning speed (Default: `228–240 PWM`).
* **Trim Offset**: Adjusts straight-line drift. If the robot drifts slightly to the left, increase the trim (+).
* **Nudge Duration**: Controls micro-nudge pulse length (Default: `250 ms`).

### Servo Pose Calibration
* Each joint has calibrated safety limits stored in `config/robot_config.yaml`:
  * **Shoulder (S1)**: `Min: 60°`, `Neutral: 93°`, `Max: 170°`.
  * **Elbow (S2)**: `Min: 0°`, `Neutral: 45°`, `Max: 45°` *(Strict safety clamp)*.
  * **Gripper (S3)**: `Close: 40°`, `Neutral: 110°`, `Open: 170°`.

### Network Mode Settings
* Under the **Network** tab in the Cockpit:
  * View available Wi-Fi networks with green **SAVED** badges.
  * Connect to any existing network with one tap without re-entering the password.
  * Switch to **Hotspot (AP)** mode whenever taking the robot to an outdoor or offsite location.
  * Your preferred default boot mode (`Wi-Fi` or `AP`) is preserved permanently.

---

## 9. Safety, Maintenance & Troubleshooting

### Safety Rules
1. **Never Force the Arm by Hand**: Moving the servos manually while powered will strip internal metal gears. Always use the web sliders.
2. **Respect the Elbow Clamp ($S2 \le 45^\circ$)**: Do not attempt to force the elbow past $45^\circ$. The mechanical linkage is designed to operate safely within this range.
3. **Keep Fingers Clear of the Claw**: The gripper has significant clamping force. Do not place fingers between the jaws during operation.
4. **Use Emergency Stop**: Tap the large red **[STOP]** button in the top navigation bar at any time to instantly cut drive power.

### Troubleshooting Quick Guide

| Symptom | Probable Cause | Instant Solution |
| :--- | :--- | :--- |
| **Cannot connect to `egrabbot.local:5001`** | Phone/PC is on the wrong Wi-Fi network. | Check your device's Wi-Fi. Ensure you are connected to `Erovoutika_Grab_Bot` or the same local Wi-Fi router. Check the robot LCD for the current IP address. |
| **Video feed is black or loading spinner spins** | Camera cable loose or service re-initializing. | Refresh the browser page. If it persists, tap **[Restart Service]** under the System Settings tab in the cockpit. |
| **Robot does not drive forward** | Battery low or motor speed set too low. | Check battery gauge. If below 20%, plug in the charger. Ensure Base Speed is at least 220 PWM to overcome floor friction. |
| **Arm does not move** | Bluetooth connection dropped. | Tap **[Reconnect Bluetooth]** in the cockpit. The robot will re-bind `/dev/rfcomm0` automatically. |
| **Robot stutters while following** | Target is moving erratically. | Hold the target still for 1 second. The robot will detect zero velocity and transition into smooth fast cruise mode. |
| **Gripper doesn't close completely** | Object is larger than claw span. | The gripper closes until it meets resistance or reaches its calibrated limit ($40^\circ$). For wider items, reposition the object sideways. |
