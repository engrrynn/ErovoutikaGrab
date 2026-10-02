#!/usr/bin/env python3
"""Interactive Gripper Sweet Spot Calibration Tool.

Run this script on the Raspberry Pi connected to the HDMI display to visually
align the gripper sweet spot box with the exact physical landing position of the gripper jaws.

Features:
- Multi-Modal Input: Touchscreen / Mouse click-and-drag, on-screen touch buttons,
  OpenCV window keyboard controls (case-insensitive & 32-bit arrow keysyms),
  and non-blocking terminal / SSH keyboard input.
- Semi-transparent touch HUD with D-pad, size adjustments, save, reset, and exit buttons.
- Real-time resolution scaling (640x480 camera to 800x480 HDMI screen).
- Saves calibrated coordinates directly to config/vision_config.yaml.
- Automatic fallback to MockCameraAdapter if camera is busy or --mock is specified.
"""

import os
import select
import sys
import threading
import time

# Auto-bind HDMI display if launched from SSH or remote terminal
if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
    os.environ["DISPLAY"] = ":0"
if not os.environ.get("XDG_RUNTIME_DIR") and os.path.exists("/run/user/1000"):
    os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
if not os.environ.get("WAYLAND_DISPLAY") and os.path.exists("/run/user/1000/wayland-0"):
    os.environ["WAYLAND_DISPLAY"] = "wayland-0"

import cv2
import numpy as np
import yaml

# Add parent directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.adapters.camera.mock_camera import MockCameraAdapter
from src.adapters.camera.v4l2_cam import V4L2CameraAdapter

CONFIG_PATH = os.path.join(BASE_DIR, "config/vision_config.yaml")

# Screen dimensions for HDMI LCD (800x480)
SCREEN_W = 800
SCREEN_H = 480

# Mouse wheel event code
MOUSEWHEEL_EVENT = getattr(cv2, "EVENT_MOUSEWHEEL", 10)

# Touch buttons layout (coordinates in 800x480 display space)
BUTTONS = [
    {"id": "left", "label": "<", "rect": (15, 422, 65, 472), "bg": (60, 60, 60), "fg": (255, 255, 255)},
    {"id": "up", "label": "^", "rect": (75, 422, 125, 472), "bg": (60, 60, 60), "fg": (255, 255, 255)},
    {"id": "down", "label": "v", "rect": (135, 422, 185, 472), "bg": (60, 60, 60), "fg": (255, 255, 255)},
    {"id": "right", "label": ">", "rect": (195, 422, 245, 472), "bg": (60, 60, 60), "fg": (255, 255, 255)},
    {"id": "w_dec", "label": "W-", "rect": (260, 422, 315, 472), "bg": (90, 50, 20), "fg": (255, 255, 255)},
    {"id": "w_inc", "label": "W+", "rect": (325, 422, 380, 472), "bg": (90, 50, 20), "fg": (255, 255, 255)},
    {"id": "h_dec", "label": "H-", "rect": (395, 422, 450, 472), "bg": (90, 50, 20), "fg": (255, 255, 255)},
    {"id": "h_inc", "label": "H+", "rect": (460, 422, 515, 472), "bg": (90, 50, 20), "fg": (255, 255, 255)},
    {"id": "reset", "label": "RESET", "rect": (525, 422, 595, 472), "bg": (60, 60, 60), "fg": (200, 200, 200)},
    {"id": "save", "label": "SAVE", "rect": (605, 422, 695, 472), "bg": (20, 140, 40), "fg": (255, 255, 255)},
    {"id": "exit", "label": "EXIT", "rect": (705, 422, 785, 472), "bg": (30, 30, 160), "fg": (255, 255, 255)},
]


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\n[Calibration] Successfully saved updated coordinates to {CONFIG_PATH}")


class CalibrationState:
    """Thread-safe state for interactive sweet-spot calibration."""

    def __init__(self, cx, cy, bw, bh, frame_w=640, frame_h=480):
        self.lock = threading.Lock()
        self.cx = cx
        self.cy = cy
        self.bw = bw
        self.bh = bh
        self.orig_cx = cx
        self.orig_cy = cy
        self.orig_bw = bw
        self.orig_bh = bh
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.step = 5

        self.save_requested = False
        self.quit_requested = False
        self.toast_msg = "Ready. Tap screen, use buttons, or press W/A/S/D"
        self.toast_time = time.time()
        self.active_button_id = None
        self.active_button_time = 0.0
        self.is_dragging = False

    def set_toast(self, msg):
        self.toast_msg = msg
        self.toast_time = time.time()

    def set_center(self, fx, fy, source="Touch"):
        with self.lock:
            # Clamp to frame
            self.cx = max(self.bw // 2, min(self.frame_w - self.bw // 2, int(fx)))
            self.cy = max(self.bh // 2, min(self.frame_h - self.bh // 2, int(fy)))
            self.set_toast(f"[{source}] Center: ({self.cx}, {self.cy})")
        print(f"\r[{source}] Position set -> Center: ({self.cx}, {self.cy})", end="", flush=True)

    def trigger_button(self, btn_id):
        self.active_button_id = btn_id
        self.active_button_time = time.time()
        self.handle_action(btn_id, source="TouchButton")

    def handle_action(self, action, source="User"):
        with self.lock:
            if action in ("up", "^"):
                self.cy = max(self.bh // 2, self.cy - self.step)
                self.set_toast(f"[{source}] UP -> ({self.cx}, {self.cy})")
            elif action in ("down", "v"):
                self.cy = min(self.frame_h - self.bh // 2, self.cy + self.step)
                self.set_toast(f"[{source}] DOWN -> ({self.cx}, {self.cy})")
            elif action in ("left", "<"):
                self.cx = max(self.bw // 2, self.cx - self.step)
                self.set_toast(f"[{source}] LEFT -> ({self.cx}, {self.cy})")
            elif action in ("right", ">"):
                self.cx = min(self.frame_w - self.bw // 2, self.cx + self.step)
                self.set_toast(f"[{source}] RIGHT -> ({self.cx}, {self.cy})")
            elif action == "w_dec":
                self.bw = max(20, self.bw - self.step)
                self.set_toast(f"[{source}] Width: {self.bw}px")
            elif action == "w_inc":
                self.bw = min(self.frame_w, self.bw + self.step)
                self.set_toast(f"[{source}] Width: {self.bw}px")
            elif action == "h_dec":
                self.bh = max(20, self.bh - self.step)
                self.set_toast(f"[{source}] Height: {self.bh}px")
            elif action == "h_inc":
                self.bh = min(self.frame_h, self.bh + self.step)
                self.set_toast(f"[{source}] Height: {self.bh}px")
            elif action == "reset":
                self.cx, self.cy = self.orig_cx, self.orig_cy
                self.bw, self.bh = self.orig_bw, self.orig_bh
                self.set_toast(f"[{source}] Reset to ({self.cx}, {self.cy})")
            elif action == "step_toggle":
                steps = [1, 2, 5, 10]
                idx = (steps.index(self.step) + 1) % len(steps) if self.step in steps else 0
                self.step = steps[idx]
                self.set_toast(f"[{source}] Step Size: {self.step}px")
            elif action == "save":
                self.save_requested = True
                self.set_toast("[SAVE] Saving calibration to YAML...")
            elif action in ("exit", "quit"):
                self.quit_requested = True
                self.set_toast("[EXIT] Quitting without saving...")

        print(f"\r[{source}] {action.upper()}: Center=({self.cx}, {self.cy}), Size={self.bw}x{self.bh}    ", end="", flush=True)


def on_mouse(event, x, y, flags, param):
    """OpenCV HighGUI mouse event callback."""
    state = param

    # 1. Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click landed on any control buttons in bottom bar
        clicked_button = None
        for btn in BUTTONS:
            bx1, by1, bx2, by2 = btn["rect"]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                clicked_button = btn["id"]
                break

        if clicked_button:
            state.trigger_button(clicked_button)
            return

        # Click landed in the camera viewport
        if y < 415:
            state.is_dragging = True
            fx = int(x * state.frame_w / SCREEN_W)
            fy = int(y * state.frame_h / SCREEN_H)
            state.set_center(fx, fy, source="Click")
            return

    # 2. Mouse drag
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.is_dragging and y < 415:
            fx = int(x * state.frame_w / SCREEN_W)
            fy = int(y * state.frame_h / SCREEN_H)
            state.set_center(fx, fy, source="Drag")

    # 3. Mouse release
    elif event == cv2.EVENT_LBUTTONUP:
        state.is_dragging = False

    # 4. Mouse wheel resizing
    elif event == MOUSEWHEEL_EVENT:
        if flags > 0:
            state.handle_action("w_inc", source="Wheel")
            state.handle_action("h_inc", source="Wheel")
        elif flags < 0:
            state.handle_action("w_dec", source="Wheel")
            state.handle_action("h_dec", source="Wheel")


def terminal_listener(state, stop_event):
    """Daemon thread reading stdin so typing in the terminal/SSH works immediately."""
    if not sys.stdin.isatty():
        # Fallback for piped or non-interactive stdin
        while not stop_event.is_set():
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not rlist:
                    continue
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd in ("w", "up"):
                    state.handle_action("up", source="Terminal")
                elif cmd in ("s", "down"):
                    state.handle_action("down", source="Terminal")
                elif cmd in ("a", "left"):
                    state.handle_action("left", source="Terminal")
                elif cmd in ("d", "right"):
                    state.handle_action("right", source="Terminal")
                elif cmd in ("save", "enter"):
                    state.handle_action("save", source="Terminal")
                    break
                elif cmd in ("q", "quit", "exit"):
                    state.handle_action("quit", source="Terminal")
                    break
                elif cmd in ("r", "reset"):
                    state.handle_action("reset", source="Terminal")
            except Exception:
                break
        return

    # Interactive TTY terminal
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
    except Exception:
        return

    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            rlist, _, _ = select.select([fd], [], [], 0.1)
            if not rlist:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                break

            # Handle ANSI escape sequence for arrow keys
            if ch == "\x1b":
                r2, _, _ = select.select([fd], [], [], 0.05)
                if r2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == "[":
                        r3, _, _ = select.select([fd], [], [], 0.05)
                        if r3:
                            ch3 = sys.stdin.read(1)
                            if ch3 == "A":
                                state.handle_action("up", source="Terminal")
                            elif ch3 == "B":
                                state.handle_action("down", source="Terminal")
                            elif ch3 == "C":
                                state.handle_action("right", source="Terminal")
                            elif ch3 == "D":
                                state.handle_action("left", source="Terminal")
                            continue
                # Lone ESC key
                state.handle_action("quit", source="Terminal")
                break

            c_low = ch.lower()
            if ch in ("\r", "\n", " "):
                state.handle_action("save", source="Terminal")
                break
            elif c_low == "q":
                state.handle_action("quit", source="Terminal")
                break
            elif c_low == "w":
                state.handle_action("up", source="Terminal")
            elif c_low == "s":
                state.handle_action("down", source="Terminal")
            elif c_low == "a":
                state.handle_action("left", source="Terminal")
            elif c_low == "d":
                state.handle_action("right", source="Terminal")
            elif ch in ("[", "{"):
                state.handle_action("w_dec", source="Terminal")
            elif ch in ("]", "}"):
                state.handle_action("w_inc", source="Terminal")
            elif ch in ("-", "_"):
                state.handle_action("h_dec", source="Terminal")
            elif ch in ("=", "+"):
                state.handle_action("h_inc", source="Terminal")
            elif c_low == "r":
                state.handle_action("reset", source="Terminal")
            elif ch in ("1", "2", "5", "0"):
                steps_map = {"1": 1, "2": 2, "5": 5, "0": 10}
                state.step = steps_map[ch]
                state.set_toast(f"[Terminal] Step size: {state.step}px")
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def main():
    cfg = load_config()
    spot = cfg.get("grasp_sweet_spot", {})
    cx = spot.get("center_x", 320)
    cy = spot.get("center_y", 380)
    bw = spot.get("box_width", 120)
    bh = spot.get("box_height", 90)

    cam_cfg = cfg.get("camera", {})
    frame_w = cam_cfg.get("width", 640)
    frame_h = cam_cfg.get("height", 480)

    use_mock = "--mock" in sys.argv
    if use_mock:
        print("[Mode] Using MockCameraAdapter (Synthetic Video)")
        cam = MockCameraAdapter(width=frame_w, height=frame_h)
    else:
        cam = V4L2CameraAdapter(
            device_index=cam_cfg.get("device_index", 0),
            width=frame_w,
            height=frame_h
        )

    if not cam.start():
        print("[Warning] Could not start V4L2 camera on /dev/video0.")
        print("[Fallback] Starting MockCameraAdapter for calibration...")
        cam = MockCameraAdapter(width=frame_w, height=frame_h)
        if not cam.start():
            print("[Error] Failed to initialize camera adapter.")
            sys.exit(1)

    state = CalibrationState(cx, cy, bw, bh, frame_w, frame_h)

    print("\n=======================================================")
    print("   🎯 ErovoutikaGrab - Camera Sweet Spot Calibration   ")
    print("=======================================================")
    print("ALL INPUT MODALITIES ARE ACTIVE:")
    print("  1. Touchscreen / Mouse:")
    print("     - Tap / Drag anywhere to place sweet spot box")
    print("     - Tap buttons at the bottom of the screen")
    print("     - Scroll wheel to expand / shrink box")
    print("  2. Window Keyboard Controls:")
    print("     - W/A/S/D or Arrow Keys : Move Sweet Spot")
    print("     - [ / ]                 : Adjust Width")
    print("     - - / =                 : Adjust Height")
    print("     - 1, 2, 5, 0            : Set Step (1, 2, 5, 10px)")
    print("     - F                     : Toggle Fullscreen")
    print("     - ENTER / SPACE         : Save and Exit")
    print("     - ESC / Q               : Quit without saving")
    print("  3. Terminal / SSH Input:")
    print("     - Type W, A, S, D, Enter, or Q right here in this terminal!")
    print("  4. Web Alternative:")
    print("     - python3 scripts/web_calibrate.py (open in browser)")
    print("=======================================================\n")

    window_name = "ErovoutikaGrab - Sweet Spot Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, on_mouse, state)

    # Start background terminal stdin thread
    stop_event = threading.Event()
    term_thread = threading.Thread(target=terminal_listener, args=(state, stop_event), daemon=True)
    term_thread.start()

    is_fullscreen = True

    try:
        while True:
            # Check exit conditions
            if state.save_requested:
                cfg.setdefault("grasp_sweet_spot", {})
                cfg["grasp_sweet_spot"]["center_x"] = int(state.cx)
                cfg["grasp_sweet_spot"]["center_y"] = int(state.cy)
                cfg["grasp_sweet_spot"]["box_width"] = int(state.bw)
                cfg["grasp_sweet_spot"]["box_height"] = int(state.bh)
                save_config(cfg)
                time.sleep(0.5)
                break

            if state.quit_requested:
                print("\n[Calibration] Exited without saving.")
                break

            frame = cam.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            vis = frame.copy()

            # Read thread-safe coordinates
            with state.lock:
                cur_cx = state.cx
                cur_cy = state.cy
                cur_bw = state.bw
                cur_bh = state.bh
                toast = state.toast_msg
                toast_age = time.time() - state.toast_time

            # 1. Draw sweet spot crosshairs and bounding box in native frame coordinates
            top_left = (cur_cx - cur_bw // 2, cur_cy - cur_bh // 2)
            bottom_right = (cur_cx + cur_bw // 2, cur_cy + cur_bh // 2)
            cv2.rectangle(vis, top_left, bottom_right, (255, 150, 65), 2)
            cv2.drawMarker(vis, (cur_cx, cur_cy), (255, 150, 65), cv2.MARKER_CROSS, 25, 2)

            # 2. Resize to 800x480 screen display resolution to eliminate letterboxing
            display_frame = cv2.resize(vis, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

            # 3. Draw semi-transparent HUD overlays (Top bar & Bottom buttons bar in Deep Royal Navy)
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (SCREEN_W, 55), (38, 20, 11), -1)
            cv2.rectangle(overlay, (0, 415), (SCREEN_W, SCREEN_H), (38, 20, 11), -1)

            # Draw button background rectangles on overlay
            now = time.time()
            for btn in BUTTONS:
                bx1, by1, bx2, by2 = btn["rect"]
                is_active = (state.active_button_id == btn["id"] and (now - state.active_button_time) < 0.2)
                bg_color = (255, 150, 65) if is_active else btn["bg"]
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)

            # Blend overlay with camera frame (0.65 alpha preserves camera visibility)
            cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0, display_frame)

            # 4. Render Top Bar HUD Text in Royal Blue
            cv2.putText(
                display_frame,
                f"SWEET SPOT: ({cur_cx}, {cur_cy})  SIZE: {cur_bw}x{cur_bh}  STEP: {state.step}px",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 150, 65),
                2,
                cv2.LINE_AA
            )
            # Display toast / instructions in Ice Royal Blue
            if toast_age < 3.0:
                cv2.putText(display_frame, toast, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (254, 219, 191), 1, cv2.LINE_AA)
            else:
                cv2.putText(
                    display_frame,
                    "Click/Drag in window to place box | Click buttons below | Type in terminal/SSH",
                    (15, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (210, 185, 160),
                    1,
                    cv2.LINE_AA
                )

            # 5. Render Crisp Button Outlines and Centered Labels
            for btn in BUTTONS:
                bx1, by1, bx2, by2 = btn["rect"]
                is_active = (state.active_button_id == btn["id"] and (now - state.active_button_time) < 0.2)
                border_color = (255, 150, 65) if is_active else (140, 70, 35)
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), border_color, 2 if is_active else 1)

                (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                tx = bx1 + (bx2 - bx1 - tw) // 2
                ty = by1 + (by2 - by1 + th) // 2
                text_color = (255, 255, 255) if is_active else btn["fg"]
                cv2.putText(display_frame, btn["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            # 6. Window Keyboard Input Handling (32-bit keysyms + normalized characters)
            raw_key = cv2.waitKeyEx(20)
            if raw_key != -1:
                k = raw_key & 0xFF
                char = chr(k).lower() if (32 <= k <= 126) else ""

                # Quit (ESC / Q)
                if raw_key in (27, 65307) or char == "q":
                    state.handle_action("quit", source="WindowKey")
                # Save (ENTER / SPACE)
                elif raw_key in (10, 13, 65293, 65421) or char == " ":
                    state.handle_action("save", source="WindowKey")
                # Fullscreen Toggle
                elif char == "f":
                    is_fullscreen = not is_fullscreen
                    prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
                    state.set_toast(f"[Window] Fullscreen: {is_fullscreen}")
                # Directional Controls (Linux keysyms + chars)
                elif raw_key in (65362, 2490368, 63232) or char == "w" or (raw_key == 82 and char != "r"):
                    state.handle_action("up", source="WindowKey")
                elif raw_key in (65364, 2621440, 63233) or char == "s" or (raw_key == 84 and char != "t"):
                    state.handle_action("down", source="WindowKey")
                elif raw_key in (65361, 2424832, 63234) or char == "a" or (raw_key == 81 and char != "q"):
                    state.handle_action("left", source="WindowKey")
                elif raw_key in (65363, 2555904, 63235) or char == "d":
                    state.handle_action("right", source="WindowKey")
                # Size Controls
                elif char in ("[", "{"):
                    state.handle_action("w_dec", source="WindowKey")
                elif char in ("]", "}"):
                    state.handle_action("w_inc", source="WindowKey")
                elif char in ("-", "_"):
                    state.handle_action("h_dec", source="WindowKey")
                elif char in ("=", "+"):
                    state.handle_action("h_inc", source="WindowKey")
                elif char == "r":
                    state.handle_action("reset", source="WindowKey")
                # Step adjustments
                elif char in ("1", "2", "5", "0"):
                    steps_map = {"1": 1, "2": 2, "5": 5, "0": 10}
                    state.step = steps_map[char]
                    state.set_toast(f"[Window] Step size: {state.step}px")

    finally:
        stop_event.set()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
