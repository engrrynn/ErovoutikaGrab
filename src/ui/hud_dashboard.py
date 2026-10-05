import os
import sys
import time
from typing import Optional
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QProgressBar, QVBoxLayout, QWidget
)

from src.core.context import RobotContext
from src.core.state_machine import AutonomousStateMachine, RobotState

try:
    import scripts.wifi_manager as wifi_manager
except ImportError:
    wifi_manager = None

LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "company_asset", "Erovoutika-Light-Logo-1.webp"))
_CACHED_HUD_LOGO: Optional[np.ndarray] = None


def get_hud_logo(target_h: int = 36) -> Optional[np.ndarray]:
    global _CACHED_HUD_LOGO
    if _CACHED_HUD_LOGO is not None and _CACHED_HUD_LOGO.shape[0] == target_h:
        return _CACHED_HUD_LOGO
    if os.path.exists(LOGO_PATH):
        raw = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            aspect = raw.shape[1] / raw.shape[0]
            target_w = int(target_h * aspect)
            _CACHED_HUD_LOGO = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return _CACHED_HUD_LOGO
    return None


class HUDDashboard(QMainWindow):
    """Real-time LCD HDMI HUD showing camera feed, VLM detection, and telemetry."""

    def __init__(self, context: RobotContext, state_machine: AutonomousStateMachine):
        super().__init__()
        self.context = context
        self.sm = state_machine

        self.setWindowTitle("ErovoutikaGrab - Autonomous Categorizer & Picker")
        self.resize(1280, 720)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b142a; color: #E0E0E0; font-family: 'DejaVu Sans', sans-serif; }
            QGroupBox { border: 1px solid #1e3a8a; border-radius: 6px; margin-top: 8px; font-weight: bold; color: #3b82f6; font-size: 13px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel { color: #CFD8DC; font-size: 12px; }
            QPushButton { background-color: #111c44; border: 1px solid #1e3a8a; border-radius: 4px; color: #F8FAFC; padding: 6px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background-color: #1e3a8a; }
            QPushButton:pressed { background-color: #2563eb; }
            QProgressBar { border: 1px solid #1e3a8a; border-radius: 4px; text-align: center; color: white; background: #070d1d; }
            QProgressBar::chunk { background-color: #3b82f6; }
        """)

        self._build_ui()

        # Update timer (30 FPS GUI refresh)
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_hud)
        self.timer.start(33)

        # Teleop key tracking and watchdog timer (prevents runaway movement on dropped release)
        self._teleop_active_keys = set()
        self._teleop_watchdog = QTimer(self)
        self._teleop_watchdog.setSingleShot(True)
        self._teleop_watchdog.timeout.connect(self._teleop_timeout_stop)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left: Live Video Feed with Overlays
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #000000; border: 2px solid #1e3a8a; border-radius: 6px;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Awaiting Video Stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        video_layout.addWidget(self.video_label)
        main_layout.addWidget(video_frame, stretch=6)

        # Right: Telemetry & Controls Panel
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(8)

        # 1. Autonomous Status Pill
        self.state_pill = QLabel("STATE: IDLE")
        self.state_pill.setAlignment(Qt.AlignCenter)
        self.state_pill.setWordWrap(True)
        self.state_pill.setStyleSheet("""
            background-color: #111c44; color: #60a5fa; font-size: 15px; font-weight: bold;
            border: 2px solid #3b82f6; border-radius: 6px; padding: 7px;
        """)
        panel_layout.addWidget(self.state_pill)

        # 2. YOLOE AI Vision & Alignment Card
        vlm_group = QGroupBox("👁️ YOLOE Vision & Alignment Telemetry")
        vlm_layout = QVBoxLayout(vlm_group)

        self.cat_name_label = QLabel("Object: Scanning for targets...")
        self.cat_name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #60a5fa;")
        self.cat_name_label.setWordWrap(True)
        vlm_layout.addWidget(self.cat_name_label)

        self.lbl_yoloe_pickable = QLabel("Pickable: --")
        self.lbl_yoloe_pickable.setWordWrap(True)
        vlm_layout.addWidget(self.lbl_yoloe_pickable)

        self.lbl_yoloe_align = QLabel("Sweet Spot: --")
        self.lbl_yoloe_align.setWordWrap(True)
        vlm_layout.addWidget(self.lbl_yoloe_align)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFormat("Confidence: %p%")
        vlm_layout.addWidget(self.conf_bar)

        self.vlm_latency_label = QLabel("Inference: --")
        self.vlm_latency_label.setWordWrap(True)
        vlm_layout.addWidget(self.vlm_latency_label)
        panel_layout.addWidget(vlm_group)

        # 3. Hardware & Comms Telemetry Card
        telem_group = QGroupBox("Microcontroller & Wireless Comms")
        telem_layout = QGridLayout(telem_group)
        telem_layout.setSpacing(6)

        self.lbl_bt_status = QLabel("BT Link: Disconnected")
        self.lbl_bt_status.setWordWrap(True)
        self.lbl_wifi_status = QLabel("WiFi: Checking...")
        self.lbl_wifi_status.setWordWrap(True)
        self.lbl_ping = QLabel("Ping: -- ms")
        self.lbl_ping.setWordWrap(True)
        self.lbl_motors = QLabel("PWM (L/R): 0 / 0")
        self.lbl_motors.setWordWrap(True)
        self.lbl_servos = QLabel("Servos (S1/S2/S3): 20° / 70° / 170°")
        self.lbl_servos.setWordWrap(True)

        self._last_wifi_check_time = 0.0
        self._cached_wifi_text = "WiFi: Checking..."
        self._cached_wifi_style = "color: #94A3B8; font-weight: bold;"

        # Dedicated rows for long wireless statuses so text is 100% visible
        telem_layout.addWidget(self.lbl_bt_status, 0, 0, 1, 2)
        telem_layout.addWidget(self.lbl_wifi_status, 1, 0, 1, 2)
        telem_layout.addWidget(self.lbl_ping, 2, 0)
        telem_layout.addWidget(self.lbl_motors, 2, 1)
        telem_layout.addWidget(self.lbl_servos, 3, 0, 1, 2)
        panel_layout.addWidget(telem_group)

        # 4. Action Controls
        ctrl_group = QGroupBox("Mission Control & Manual Override")
        ctrl_layout = QGridLayout(ctrl_group)

        self.btn_auto = QPushButton("▶ START AUTO PICK")
        self.btn_auto.setStyleSheet("background-color: #065F46; border: 1px solid #059669; color: white;")
        self.btn_auto.clicked.connect(self._toggle_autonomous)
        ctrl_layout.addWidget(self.btn_auto, 0, 0, 1, 2)

        self.btn_stop = QPushButton("⏹ EMERGENCY STOP")
        self.btn_stop.setStyleSheet("background-color: #991B1B; border: 1px solid #DC2626; color: white;")
        self.btn_stop.clicked.connect(self._emergency_stop)
        ctrl_layout.addWidget(self.btn_stop, 1, 0, 1, 2)

        # Automated Showcase Button
        self.btn_showcase = QPushButton("★ SHOWCASE CAPABILITIES")
        self.btn_showcase.setStyleSheet("background-color: #1d4ed8; border: 1px solid #3b82f6; color: white; font-weight: bold; padding: 6px;")
        self.btn_showcase.clicked.connect(self._toggle_showcase)
        ctrl_layout.addWidget(self.btn_showcase, 2, 0, 1, 2)

        # Manual Arm Buttons
        btn_pick = QPushButton("Pick Sequence")
        btn_pick.clicked.connect(lambda: self.sm.arm.execute_pick_sequence())
        btn_stow = QPushButton("Stow Arm")
        btn_stow.clicked.connect(lambda: self.sm.arm.stow())
        btn_open = QPushButton("Open Gripper")
        btn_open.clicked.connect(lambda: self.sm.arm.open_gripper())
        btn_close = QPushButton("Close Gripper")
        btn_close.clicked.connect(lambda: self.sm.arm.close_gripper())

        ctrl_layout.addWidget(btn_pick, 3, 0)
        ctrl_layout.addWidget(btn_stow, 3, 1)
        ctrl_layout.addWidget(btn_open, 4, 0)
        ctrl_layout.addWidget(btn_close, 4, 1)

        panel_layout.addWidget(ctrl_group)
        panel_layout.addStretch()

        main_layout.addWidget(panel, stretch=3)

    def _toggle_showcase(self):
        """Launches the automated capability showcase in a background worker thread."""
        if getattr(self, "_showcase_running", False):
            return
        self._showcase_running = True
        self.btn_showcase.setText("★ SHOWCASE IN PROGRESS...")
        self.btn_showcase.setStyleSheet("background-color: #854D0E; border: 1px solid #F59E0B; color: white; font-weight: bold; padding: 6px;")

        import threading
        import time
        def worker():
            try:
                arm = self.sm.arm
                if arm:
                    arm.center()
                    time.sleep(0.4)
                    arm.open_gripper()
                    time.sleep(0.4)
                    arm.arm_down()
                    time.sleep(0.6)
                    arm.close_gripper()
                    time.sleep(0.4)
                    arm.stow()
                    time.sleep(0.6)
                    arm.open_gripper()
                    time.sleep(0.4)
                    arm.center()
            except Exception as e:
                print(f"[HUD] Showcase worker notice: {e}")
            finally:
                self._showcase_running = False
                self.btn_showcase.setText("★ SHOWCASE CAPABILITIES")
                self.btn_showcase.setStyleSheet("background-color: #1d4ed8; border: 1px solid #3b82f6; color: white; font-weight: bold; padding: 6px;")

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _toggle_autonomous(self):
        if self.sm.state == RobotState.IDLE:
            self.sm.start_autonomous_run()
            self.btn_auto.setText("⏸ PAUSE AUTO")
            self.btn_auto.setStyleSheet("background-color: #854D0E; border: 1px solid #CA8A04;")
        else:
            self.sm.stop()
            self.btn_auto.setText("▶ START AUTO PICK")
            self.btn_auto.setStyleSheet("background-color: #065F46; border: 1px solid #059669;")

    def _emergency_stop(self):
        if hasattr(self, "_teleop_active_keys"):
            self._teleop_active_keys.clear()
        if hasattr(self, "_teleop_watchdog"):
            self._teleop_watchdog.stop()
        self.sm.stop()
        self.btn_auto.setText("▶ START AUTO PICK")
        self.btn_auto.setStyleSheet("background-color: #065F46; border: 1px solid #059669;")

    def _teleop_timeout_stop(self):
        """Safety watchdog callback: halts robot if key refresh stops."""
        if hasattr(self, "_teleop_active_keys"):
            self._teleop_active_keys.clear()
        self.sm.comm.send_stop()

    def _send_drive_for_keys(self):
        """Sends drive command corresponding to currently active keyboard controls."""
        motors_cfg = {}
        if hasattr(self.context, "config") and isinstance(self.context.config, dict):
            motors_cfg = self.context.config.get("robot", {}).get("motors", {})
        base_speed = int(motors_cfg.get("base_speed", 225))
        turn_speed = int(motors_cfg.get("turn_speed", 240))
        trim = int(motors_cfg.get("trim_offset", 0))

        left = base_speed
        right = base_speed
        if trim > 0:
            right = max(100, right - trim)
        elif trim < 0:
            left = max(100, left - abs(trim))

        if Qt.Key_W in self._teleop_active_keys:
            self.sm.comm.send_drive(left, right)
        elif Qt.Key_S in self._teleop_active_keys:
            self.sm.comm.send_drive(-left, -right)
        elif Qt.Key_A in self._teleop_active_keys:
            self.sm.comm.send_drive(-turn_speed, turn_speed)
        elif Qt.Key_D in self._teleop_active_keys:
            self.sm.comm.send_drive(turn_speed, -turn_speed)

    def keyPressEvent(self, event):
        """Keyboard manual controls (W/A/S/D drive, Space stop) using calibrated motor parameters."""
        if event.isAutoRepeat():
            return

        k = event.key()
        if k in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D):
            self._teleop_active_keys.add(k)
            self._send_drive_for_keys()
            self._teleop_watchdog.start(350)
        elif k == Qt.Key_Space or k == Qt.Key_Q:
            self._emergency_stop()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Immediately halts robot on key release unless another direction key is held."""
        if event.isAutoRepeat():
            return

        k = event.key()
        if k in (Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D):
            self._teleop_active_keys.discard(k)
            if self._teleop_active_keys:
                self._send_drive_for_keys()
                self._teleop_watchdog.start(350)
            else:
                self._teleop_watchdog.stop()
                self.sm.comm.send_stop()
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        """Stops motion immediately if dashboard loses window focus."""
        if hasattr(self, "_teleop_active_keys"):
            self._teleop_active_keys.clear()
        if hasattr(self, "_teleop_watchdog"):
            self._teleop_watchdog.stop()
        self.sm.comm.send_stop()
        super().focusOutEvent(event)

    def _update_hud(self):
        # 1. Step state machine
        self.sm.step()

        # 2. Update telemetry labels
        telem = self.context.get_telemetry()
        if telem.connected:
            self.lbl_bt_status.setText("BT Link: Connected (HC-05)")
            self.lbl_bt_status.setStyleSheet("color: #4ADE80; font-weight: bold;")
            self.lbl_ping.setText(f"Ping: {telem.ping_ms} ms")
        else:
            self.lbl_bt_status.setText("BT Link: Disconnected")
            self.lbl_bt_status.setStyleSheet("color: #F87171; font-weight: bold;")
            self.lbl_ping.setText("Ping: -- ms")

        # Update Wi-Fi Telemetry joined with BT
        now = time.time()
        if wifi_manager and (now - self._last_wifi_check_time > 1.5):
            self._last_wifi_check_time = now
            try:
                wstatus = wifi_manager.get_wifi_status(cached=True)
                wmode = (wstatus.get("mode") or "disconnected").lower()
                wssid = wstatus.get("ssid") or "None"
                wip = wstatus.get("ip") or ""
                if wmode == "hotspot":
                    self._cached_wifi_text = f"WiFi: 📡 AP ({wssid}) | {wip}" if wip else f"WiFi: 📡 AP ({wssid})"
                    self._cached_wifi_style = "color: #F59E0B; font-weight: bold;"
                elif wmode == "client":
                    self._cached_wifi_text = f"WiFi: 📶 {wssid} | {wip}" if wip else f"WiFi: 📶 {wssid}"
                    self._cached_wifi_style = "color: #4ADE80; font-weight: bold;"
                else:
                    is_auto_ap = getattr(wifi_manager, "_auto_ap_in_progress", False)
                    if is_auto_ap:
                        self._cached_wifi_text = "WiFi: 📡 Restoring AP Mode..."
                        self._cached_wifi_style = "color: #F59E0B; font-weight: bold;"
                    else:
                        self._cached_wifi_text = "WiFi: Offline (Auto-AP checking...)"
                        self._cached_wifi_style = "color: #F87171; font-weight: bold;"
                    if hasattr(wifi_manager, "check_auto_ap_fallback"):
                        wifi_manager.check_auto_ap_fallback(threshold_checks=2)
            except Exception:
                pass
        self.lbl_wifi_status.setText(self._cached_wifi_text)
        self.lbl_wifi_status.setStyleSheet(self._cached_wifi_style)

        self.lbl_motors.setText(f"PWM (L/R): {telem.left_pwm} / {telem.right_pwm}")
        self.lbl_servos.setText(f"Servos: S1={telem.s1_shoulder}° S2={telem.s2_elbow}° S3={telem.s3_gripper}°")

        # 3. Update YOLOE Categorization & Alignment Card
        det = self.context.get_detection()
        if det and getattr(det, "detected", False):
            self.cat_name_label.setText(f"Object: {det.category.upper()}")
            self.cat_name_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #4ade80;")

            if getattr(det, "pickable", False):
                self.lbl_yoloe_pickable.setText("Pickable: YES (Valid Target)")
                self.lbl_yoloe_pickable.setStyleSheet("color: #4ade80; font-weight: bold;")
            else:
                self.lbl_yoloe_pickable.setText("Pickable: NO (Rejected/Too Large)")
                self.lbl_yoloe_pickable.setStyleSheet("color: #f87171; font-weight: bold;")

            sx = self.context.sweet_spot_x
            sy = self.context.sweet_spot_y
            sw = self.context.sweet_spot_w
            sh = self.context.sweet_spot_h
            cx, cy = det.center
            in_x = (sx - sw // 2) <= cx <= (sx + sw // 2)
            in_y = (sy - sh // 2) <= cy <= (sy + sh // 2)
            dx = cx - sx
            dy = sy - cy

            if in_x and in_y:
                self.lbl_yoloe_align.setText("Sweet Spot: ALIGNED [READY]")
                self.lbl_yoloe_align.setStyleSheet("color: #38bdf8; font-weight: bold;")
            else:
                self.lbl_yoloe_align.setText(f"Sweet Spot: ALIGNING (dx={dx:+d}px, dy={dy:+d}px)")
                self.lbl_yoloe_align.setStyleSheet("color: #fbbf24;")

            self.conf_bar.setValue(int(det.confidence * 100))
        else:
            self.cat_name_label.setText("Object: Scanning for targets...")
            self.cat_name_label.setStyleSheet("font-size: 14px; font-weight: normal; color: #94a3b8;")
            self.lbl_yoloe_pickable.setText("Pickable: --")
            self.lbl_yoloe_pickable.setStyleSheet("color: #94a3b8;")
            self.lbl_yoloe_align.setText("Sweet Spot: Awaiting Target")
            self.lbl_yoloe_align.setStyleSheet("color: #94a3b8;")
            self.conf_bar.setValue(0)

        if self.context.inference_latency_s > 0:
            fps = 1.0 / self.context.inference_latency_s if self.context.inference_latency_s > 0 else 0
            self.vlm_latency_label.setText(f"Inference: {self.context.inference_latency_s * 1000:.0f}ms (~{fps:.1f} FPS)")

        # 4. Update State Badge
        state_str = self.context.get_state()
        self.state_pill.setText(f"STATE: {state_str}")

        # 5. Render Video Frame with HUD Overlays
        frame = self.context.get_frame()
        if frame is not None:
            annotated = self._render_overlays(frame)
            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pix)

    def _render_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draws sweet spot box, tracking box, and alignment vectors directly on video frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        sx = self.context.sweet_spot_x
        sy = self.context.sweet_spot_y
        sw = self.context.sweet_spot_w
        sh = self.context.sweet_spot_h

        # Draw Gripper Grasp Sweet Spot Box (cyan/green)
        top_left = (sx - sw // 2, sy - sh // 2)
        bottom_right = (sx + sw // 2, sy + sh // 2)
        cv2.rectangle(vis, top_left, bottom_right, (0, 255, 255), 2)
        cv2.drawMarker(vis, (sx, sy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis, "GRIPPER SWEET SPOT", (top_left[0], top_left[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Draw Active Detection (YOLOE Detection Box takes precedence over ROI tracker)
        det = self.context.get_detection()
        if det and getattr(det, "detected", False) and getattr(det, "bounding_box", None):
            ymin, xmin, ymax, xmax = det.bounding_box
            pickable = getattr(det, "pickable", False)
            box_col = (0, 255, 0) if pickable else (0, 165, 255)

            # Draw segmentation mask polygon if present
            poly = getattr(det, "mask_polygon", None)
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], box_col)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.polylines(vis, [pts], isClosed=True, color=box_col, thickness=2)

            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), box_col, 2)
            txt_x = max(10, min(w - 220, xmin))
            cv2.putText(vis, f"{det.category} ({det.confidence:.2f})", (txt_x, max(22, ymin - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_col, 2)

            cx, cy = det.center
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            # Line connecting the center crosshair with the detected object (centered in X axis only)
            dx = cx - sx
            is_x_centered = abs(dx) <= 20
            link_col = (0, 255, 0) if is_x_centered else (0, 165, 255)
            cv2.circle(vis, (sx, sy), 5, link_col, -1)
            cv2.line(vis, (sx, sy), (cx, cy), link_col, 2)
            align_txt = f"dx={dx:+d}px [X-CENTERED]" if is_x_centered else f"dx={dx:+d}px [{'TURN RIGHT' if dx > 0 else 'TURN LEFT'}]"
            align_x = max(10, min(w - 240, xmin))
            cv2.putText(vis, align_txt, (align_x, min(h - 10, ymax + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, link_col, 2)
        elif self.context.get_tracked_roi():
            rx, ry, rw, rh = self.context.get_tracked_roi()
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cx = rx + rw // 2
            cy = ry + rh // 2
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            # Line connecting the center crosshair with the tracked object (centered in X axis only)
            dx = cx - sx
            is_x_centered = abs(dx) <= 20
            link_col = (0, 255, 0) if is_x_centered else (0, 165, 255)
            cv2.circle(vis, (sx, sy), 5, link_col, -1)
            cv2.line(vis, (sx, sy), (cx, cy), link_col, 2)
            align_txt = f"dx={dx:+d}px [X-CENTERED]" if is_x_centered else f"dx={dx:+d}px [{'TURN R' if dx > 0 else 'TURN L'}]"
            cv2.putText(vis, align_txt, (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, link_col, 2)

        # Overlay company logo at top-left
        logo = get_hud_logo(36)
        if logo is not None:
            lh, lw = logo.shape[:2]
            lx, ly = 12, 10
            if ly + lh <= h and lx + lw <= w:
                roi = vis[ly:ly+lh, lx:lx+lw]
                if logo.shape[2] == 4:
                    alpha = logo[:, :, 3].astype(float) / 255.0
                    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                    bgr = logo[:, :, :3]
                    vis[ly:ly+lh, lx:lx+lw] = (bgr * alpha + roi * (1.0 - alpha)).astype(np.uint8)
                else:
                    vis[ly:ly+lh, lx:lx+lw] = logo[:, :, :3]

        return vis


