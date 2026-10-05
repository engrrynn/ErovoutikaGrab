#!/usr/bin/env python3
"""E-GrabBot Wireless & Hotspot Management Subsystem.

Provides bidirectional switching between Access Point (Hotspot) Mode and
WiFi Client Station Mode, network scanning, credential management, and
startup mode enforcement via NetworkManager (nmcli).
"""

import os
import re
import subprocess
import sys
import threading
import time
from typing import Dict, List, Any, Optional
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_CONFIG_PATH = os.path.join(BASE_DIR, "config/network_config.yaml")

_transition_lock = threading.Lock()
_last_transition_result: Dict[str, Any] = {"status": "idle", "message": ""}


def get_transition_status() -> Dict[str, Any]:
    """Returns the latest asynchronous network transition status and details."""
    with _transition_lock:
        return dict(_last_transition_result)


def get_active_wifi_connection() -> Optional[str]:
    """Returns the name of the currently active connection on wlan0, if any."""
    try:
        proc = run_cmd(["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"], timeout_s=4.0)
        for line in proc.stdout.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 2 and parts[1] == "wlan0":
                return parts[0]
    except Exception:
        pass
    return None


def load_network_config() -> Dict[str, Any]:
    """Loads network configuration from YAML or returns defaults."""
    defaults = {
        "startup_mode": "ap",
        "ap_ssid": "egrabbot-ap",
        "ap_password": "egrabbot1234",
        "last_connected_ssid": "GFiber_ef820",
    }
    if os.path.exists(NETWORK_CONFIG_PATH):
        try:
            with open(NETWORK_CONFIG_PATH, "r") as f:
                data = yaml.safe_load(f) or {}
                defaults.update(data)
        except Exception as e:
            print(f"[WiFiManager] Warning reading config: {e}")
    return defaults


def save_network_config(cfg: Dict[str, Any]) -> bool:
    """Persists network configuration to YAML."""
    try:
        os.makedirs(os.path.dirname(NETWORK_CONFIG_PATH), exist_ok=True)
        with open(NETWORK_CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"[WiFiManager] Error saving config: {e}")
        return False


def run_cmd(cmd_list: List[str], timeout_s: float = 12.0) -> subprocess.CompletedProcess:
    """Executes system command with timeout and returns CompletedProcess."""
    return subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False
    )


def get_current_ip() -> str:
    """Retrieves current IP address on wlan0 or primary route."""
    try:
        proc = run_cmd(["hostname", "-I"], timeout_s=3.0)
        ips = proc.stdout.strip().split()
        if ips:
            # Filter for wlan0 IP if possible
            for ip in ips:
                if not ip.startswith("127."):
                    return ip
            return ips[0]
    except Exception:
        pass
    return "Disconnected"


_cached_wifi_status: Optional[Dict[str, Any]] = None
_last_wifi_status_time: float = 0.0
_wifi_status_lock = threading.Lock()


def get_wifi_status(cached: bool = True) -> Dict[str, Any]:
    """Inspects live wireless device status via nmcli with optional 2-second caching."""
    global _cached_wifi_status, _last_wifi_status_time
    now = time.time()
    if cached and _cached_wifi_status is not None and (now - _last_wifi_status_time < 2.0):
        with _wifi_status_lock:
            return dict(_cached_wifi_status)

    res = {
        "mode": "disconnected",
        "ssid": "None",
        "ip": get_current_ip(),
        "signal": 0,
        "security": "None",
        "device": "wlan0",
        "startup_mode": load_network_config().get("startup_mode", "ap"),
        "transition": get_transition_status()
    }

    try:
        # Check active connections
        proc = run_cmd(["nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "con", "show", "--active"])
        for line in proc.stdout.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 3 and ("wireless" in parts[1].lower() or "wifi" in parts[1].lower()):
                con_name = parts[0]
                if "hotspot" in con_name.lower() or "ap" in con_name.lower():
                    res["mode"] = "hotspot"
                    res["ssid"] = load_network_config().get("ap_ssid", "Erovoutika_Grab_Bot")
                else:
                    res["mode"] = "client"
                    res["ssid"] = con_name
                break

        # If in client mode, get signal strength and security
        if res["mode"] == "client":
            sig_proc = run_cmd(["nmcli", "-t", "-f", "IN-USE,SSID,SIGNAL,SECURITY", "dev", "wifi", "list"])
            for line in sig_proc.stdout.splitlines():
                if line.startswith("*"):
                    fields = line.split(":")
                    if len(fields) >= 4:
                        try:
                            res["signal"] = int(fields[2])
                        except ValueError:
                            res["signal"] = 50
                        res["security"] = fields[3] or "Open"
                    break
    except Exception as e:
        print(f"[WiFiManager] Status error: {e}")

    with _wifi_status_lock:
        _cached_wifi_status = dict(res)
        _last_wifi_status_time = now

    return res


def get_saved_wifi_connections() -> List[str]:
    """Returns list of saved wireless connection profile names (excluding AP hotspot)."""
    saved = []
    try:
        proc = run_cmd(["nmcli", "-t", "-f", "NAME,TYPE", "con", "show"])
        for line in proc.stdout.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 2 and ("wireless" in parts[1].lower() or "wifi" in parts[1].lower()):
                name = parts[0]
                if name != "egrabbot-hotspot":
                    saved.append(name)
    except Exception:
        pass
    return saved


def scan_wifi_networks() -> List[Dict[str, Any]]:
    """Scans for nearby wireless networks and returns deduplicated list sorted by signal with saved status."""
    networks = {}
    saved_list = get_saved_wifi_connections()
    try:
        proc = run_cmd(["sudo", "nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,BARS", "dev", "wifi", "list"], timeout_s=8.0)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) >= 3:
                ssid = parts[0].strip()
                if not ssid:
                    continue  # Ignore hidden SSIDs
                try:
                    sig = int(parts[1])
                except ValueError:
                    sig = 0
                sec = parts[2].strip() or "Open"
                bars = parts[3].strip() if len(parts) > 3 else "--"

                # Keep highest signal strength entry per SSID
                if ssid not in networks or sig > networks[ssid]["signal"]:
                    networks[ssid] = {
                        "ssid": ssid,
                        "signal": sig,
                        "security": sec,
                        "bars": bars,
                        "is_saved": ssid in saved_list
                    }
    except Exception as e:
        print(f"[WiFiManager] Scan error: {e}")

    # Also include any saved networks even if not currently in immediate beacon scan
    for s in saved_list:
        if s not in networks:
            networks[s] = {
                "ssid": s,
                "signal": 0,
                "security": "Saved Connection",
                "bars": "--",
                "is_saved": True
            }

    res = list(networks.values())
    res.sort(key=lambda x: (x.get("is_saved", False), x["signal"]), reverse=True)
    return res


def configure_wifi_profile(ssid: str, password: Optional[str] = None) -> bool:
    """Pre-configures a clean NetworkManager connection profile for the target SSID.
    
    Ensures correct 802-11-wireless-security settings, auto-connect flag, and interface binding
    before any radio transition occurs.
    """
    if not ssid or not ssid.strip():
        return False
    clean_ssid = ssid.strip()
    clean_pwd = password.strip() if (password and password.strip()) else None

    saved = get_saved_wifi_connections()
    if clean_ssid in saved:
        run_cmd(["sudo", "nmcli", "con", "delete", clean_ssid], timeout_s=5.0)

    add_cmd = [
        "sudo", "nmcli", "con", "add",
        "type", "wifi",
        "ifname", "wlan0",
        "con-name", clean_ssid,
        "ssid", clean_ssid,
        "connection.autoconnect", "yes"
    ]
    if clean_pwd:
        add_cmd.extend([
            "802-11-wireless-security.key-mgmt", "wpa-psk",
            "802-11-wireless-security.psk", clean_pwd
        ])
    add_proc = run_cmd(add_cmd, timeout_s=8.0)
    return add_proc.returncode == 0


def _async_transition_worker(ssid: str, password: Optional[str], was_hotspot: bool, prev_con: Optional[str]):
    """Background worker that performs the radio transition to Station mode with automatic rollback watchdog."""
    global _last_transition_result
    with _transition_lock:
        _last_transition_result = {
            "status": "in_progress",
            "target_ssid": ssid,
            "from_ap": was_hotspot,
            "start_time": time.time(),
            "message": f"Connecting to '{ssid}'..."
        }

    # Grace period (1.5 seconds) allowing HTTP response to cleanly flush to browser socket
    time.sleep(1.5)

    print(f"[WiFiManager] Starting async transition to '{ssid}' (was_hotspot={was_hotspot}, prev_con={prev_con})...")

    # Step 1: Disconnect current radio link on wlan0 cleanly
    if was_hotspot:
        run_cmd(["sudo", "nmcli", "con", "down", "egrabbot-hotspot"], timeout_s=6.0)
    elif prev_con and prev_con != ssid:
        run_cmd(["sudo", "nmcli", "con", "down", prev_con], timeout_s=6.0)

    time.sleep(1.0)

    # Step 2: Attempt connection to target SSID
    up_proc = run_cmd(["sudo", "nmcli", "con", "up", ssid], timeout_s=22.0)
    if up_proc.returncode != 0 and password:
        # Fallback via direct nmcli dev wifi connect
        cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", password]
        run_cmd(cmd, timeout_s=22.0)

    # Step 3: Poll for active status & valid DHCP IP (up to 12s)
    success = False
    new_ip = "Disconnected"
    for _ in range(12):
        time.sleep(1.0)
        active = get_active_wifi_connection()
        ip = get_current_ip()
        if active == ssid and ip not in ("Disconnected", "10.42.0.1") and not ip.startswith("127."):
            success = True
            new_ip = ip
            break

    if success:
        print(f"[WiFiManager] Successfully connected to '{ssid}'! IP: {new_ip}")
        cfg = load_network_config()
        cfg["last_connected_ssid"] = ssid
        cfg["startup_mode"] = "wifi"
        save_network_config(cfg)
        with _transition_lock:
            _last_transition_result = {
                "status": "ok",
                "target_ssid": ssid,
                "ip": new_ip,
                "message": f"Successfully connected to '{ssid}' (IP: {new_ip})"
            }
    else:
        err_msg = up_proc.stderr.strip() or up_proc.stdout.strip() or "Connection timed out or network unavailable."
        print(f"[WiFiManager] Failed to connect to '{ssid}': {err_msg}. Triggering automatic rollback watchdog...")

        with _transition_lock:
            _last_transition_result = {
                "status": "rollback",
                "target_ssid": ssid,
                "error": err_msg,
                "message": f"Failed to connect to '{ssid}'. Rolling back network."
            }

        # AUTOMATIC ROLLBACK WATCHDOG:
        if was_hotspot:
            print("[WiFiManager] Watchdog restoring Access Point Hotspot mode...")
            activate_hotspot()
            print("[WiFiManager] Access Point Hotspot restored.")
        elif prev_con:
            print(f"[WiFiManager] Watchdog restoring previous connection '{prev_con}'...")
            rb = run_cmd(["sudo", "nmcli", "con", "up", prev_con], timeout_s=12.0)
            if rb.returncode != 0:
                print(f"[WiFiManager] Previous connection '{prev_con}' also failed. Auto-fallback to AP Hotspot mode...")
                activate_hotspot()
        else:
            print("[WiFiManager] No previous connection. Auto-fallback to AP Hotspot mode...")
            activate_hotspot()


def connect_to_wifi(ssid: str, password: Optional[str] = None) -> Dict[str, Any]:
    """Seamlessly connects the robot to a specified WiFi network.
    
    If currently in AP (Hotspot) mode, pre-configures credentials and immediately
    returns transition instructions to prevent ERR_CONNECTION_RESET, then runs
    the network switch in a background thread guarded by an automatic rollback watchdog.
    """
    if not ssid or not ssid.strip():
        return {"status": "error", "message": "SSID cannot be empty."}

    ssid = ssid.strip()
    password = password.strip() if (password and password.strip()) else None

    active_con = get_active_wifi_connection()
    is_hotspot = (active_con == "egrabbot-hotspot")

    # Pre-configure profile cleanly in NetworkManager before dropping anything
    ok = configure_wifi_profile(ssid, password)
    if not ok:
        return {"status": "error", "message": f"Failed to save profile for '{ssid}' in NetworkManager."}

    # Persist target config immediately
    cfg = load_network_config()
    cfg["last_connected_ssid"] = ssid
    cfg["startup_mode"] = "wifi"
    save_network_config(cfg)

    # Launch background transition worker with rollback watchdog
    t = threading.Thread(
        target=_async_transition_worker,
        args=(ssid, password, is_hotspot, active_con),
        daemon=True
    )
    t.start()

    if is_hotspot:
        return {
            "status": "switching",
            "from_ap": True,
            "target_ssid": ssid,
            "message": (
                f"WiFi credentials for '{ssid}' saved! The robot is transitioning from Hotspot "
                f"to WiFi Station mode. Your device will now disconnect from 'Erovoutika_Grab_Bot'. "
                f"Please connect your device to '{ssid}' and open https://egrabbot.local:5001. "
                f"If connection fails, the robot will automatically restore the hotspot in 25 seconds."
            )
        }
    else:
        return {
            "status": "switching",
            "from_ap": False,
            "target_ssid": ssid,
            "message": f"Switching to '{ssid}'... Please wait."
        }


def activate_hotspot(ssid: Optional[str] = None, password: Optional[str] = None) -> Dict[str, Any]:
    """Seamlessly activates the E-GrabBot Access Point (Hotspot) on wlan0."""
    cfg = load_network_config()
    ap_ssid = ssid.strip() if ssid else cfg.get("ap_ssid", "Erovoutika_Grab_Bot")
    ap_pwd = password.strip() if password else cfg.get("ap_password", "egrabbot1234")

    # Seamless switch: Disconnect active wifi client connection on wlan0 first
    saved = get_saved_wifi_connections()
    for con in saved:
        try:
            run_cmd(["sudo", "nmcli", "con", "down", con], timeout_s=4.0)
        except Exception:
            pass
    # Cleanly disconnect active connection on wlan0
    run_cmd(["sudo", "nmcli", "dev", "disconnect", "wlan0"], timeout_s=5.0)

    # Ensure profile exists with correct settings
    chk = run_cmd(["nmcli", "con", "show", "egrabbot-hotspot"])
    if chk.returncode != 0:
        add_cmd = [
            "sudo", "nmcli", "con", "add", "type", "wifi", "ifname", "wlan0",
            "con-name", "egrabbot-hotspot", "autoconnect", "no",
            "ssid", ap_ssid, "802-11-wireless.mode", "ap", "802-11-wireless.band", "bg",
            "802-11-wireless-security.key-mgmt", "wpa-psk",
            "802-11-wireless-security.psk", ap_pwd,
            "ipv4.method", "shared"
        ]
        run_cmd(add_cmd, timeout_s=8.0)
    else:
        run_cmd(["sudo", "nmcli", "con", "modify", "egrabbot-hotspot", "802-11-wireless.ssid", ap_ssid], timeout_s=4.0)
        run_cmd(["sudo", "nmcli", "con", "modify", "egrabbot-hotspot", "802-11-wireless-security.key-mgmt", "wpa-psk"], timeout_s=4.0)
        run_cmd(["sudo", "nmcli", "con", "modify", "egrabbot-hotspot", "802-11-wireless-security.psk", ap_pwd], timeout_s=4.0)

    # Bring up the hotspot
    up_proc = run_cmd(["sudo", "nmcli", "con", "up", "egrabbot-hotspot"], timeout_s=15.0)
    if up_proc.returncode == 0:
        cfg["ap_ssid"] = ap_ssid
        cfg["ap_password"] = ap_pwd
        # CRITICAL: Preserve startup_mode (do NOT overwrite ap/wifi boot preference)
        save_network_config(cfg)
        return {
            "status": "ok",
            "message": f"Hotspot '{ap_ssid}' is active! Connect to '{ap_ssid}' (pass: '{ap_pwd}') and open https://10.42.0.1:5001",
            "ssid": ap_ssid,
            "ip": get_current_ip()
        }
    else:
        err = up_proc.stderr.strip() or up_proc.stdout.strip()
        return {"status": "error", "message": f"Failed to activate hotspot: {err}"}


def set_startup_mode(mode: str) -> Dict[str, Any]:
    """Updates preferred boot mode ('ap' or 'wifi')."""
    mode = mode.lower().strip()
    if mode not in ("ap", "wifi"):
        return {"status": "error", "message": "Mode must be 'ap' or 'wifi'."}

    cfg = load_network_config()
    cfg["startup_mode"] = mode
    save_network_config(cfg)
    return {"status": "ok", "startup_mode": mode}


def init_startup_network() -> Dict[str, Any]:
    """Initializes wireless networking on startup with robust fail-safe.
    
    If configured for AP mode, starts the Access Point Hotspot immediately.
    If configured for WiFi client mode:
      1. Checks if NetworkManager already connected to an authorized network on boot.
      2. If not, attempts to bring up last_connected_ssid or other saved profiles.
      3. Verifies a valid IP lease.
    FAIL-SAFE: If the robot does not connect to any WiFi network or fails,
    it automatically falls back to Access Point (AP) Hotspot mode by default.
    """
    cfg = load_network_config()
    target_mode = cfg.get("startup_mode", "ap")
    print(f"[WiFiManager] Initializing network (startup_mode='{target_mode}')...")

    if target_mode == "ap":
        res = activate_hotspot(cfg.get("ap_ssid"), cfg.get("ap_password"))
        print(f"[WiFiManager] AP Hotspot mode initialized: {res}")
        return res

    # Check if NetworkManager already connected to WiFi on its own
    active_con = get_active_wifi_connection()
    current_ip = get_current_ip()
    if active_con and active_con != "egrabbot-hotspot" and current_ip not in ("Disconnected", "10.42.0.1") and not current_ip.startswith("127."):
        print(f"[WiFiManager] Already connected to WiFi '{active_con}' on boot! IP: {current_ip}")
        cfg["last_connected_ssid"] = active_con
        save_network_config(cfg)
        return {"status": "ok", "mode": "client", "ssid": active_con, "ip": current_ip}

    # Attempt connection to last_connected_ssid
    last_ssid = cfg.get("last_connected_ssid")
    wifi_connected = False
    connected_ip = ""

    if last_ssid:
        print(f"[WiFiManager] Attempting boot connection to saved WiFi: '{last_ssid}'...")
        up_proc = run_cmd(["sudo", "nmcli", "con", "up", last_ssid], timeout_s=15.0)
        if up_proc.returncode == 0:
            for _ in range(6):
                time.sleep(1.0)
                ip = get_current_ip()
                if ip not in ("Disconnected", "10.42.0.1") and not ip.startswith("127."):
                    wifi_connected = True
                    connected_ip = ip
                    break

    # If last_ssid failed, try other saved connections
    if not wifi_connected:
        saved_list = get_saved_wifi_connections()
        for s in saved_list:
            if s != last_ssid:
                print(f"[WiFiManager] Trying alternate saved WiFi profile: '{s}'...")
                up = run_cmd(["sudo", "nmcli", "con", "up", s], timeout_s=12.0)
                if up.returncode == 0:
                    for _ in range(6):
                        time.sleep(1.0)
                        ip = get_current_ip()
                        if ip not in ("Disconnected", "10.42.0.1") and not ip.startswith("127."):
                            wifi_connected = True
                            connected_ip = ip
                            last_ssid = s
                            cfg["last_connected_ssid"] = s
                            save_network_config(cfg)
                            break
                if wifi_connected:
                    break

    if wifi_connected:
        print(f"[WiFiManager] Successfully connected to WiFi '{last_ssid}' on boot! IP: {connected_ip}")
        return {"status": "ok", "mode": "client", "ssid": last_ssid, "ip": connected_ip}

    # FAIL-SAFE FALLBACK: No WiFi connected -> Default to AP Hotspot mode!
    print("[WiFiManager] ⚠️ FAIL-SAFE TRIGGERED: Robot failed to connect to any WiFi network on boot.")
    print("[WiFiManager] Automatically activating Access Point (AP) Hotspot mode by default...")
    res = activate_hotspot(cfg.get("ap_ssid"), cfg.get("ap_password"))
    res["fail_safe_triggered"] = True
    print(f"[WiFiManager] Fail-safe AP Hotspot activated: {res}")
    return res


_offline_consecutive_checks = 0
_auto_ap_in_progress = False
_wifi_watchdog_started = False


def check_auto_ap_fallback(threshold_checks: int = 2) -> bool:
    """Checks if WiFi cannot connect / is offline, and automatically activates AP mode after checking.
    
    If the HUD or system detects the connection is offline or cannot connect,
    it automatically triggers Access Point (AP Hotspot) mode so the robot remains accessible.
    """
    global _offline_consecutive_checks, _auto_ap_in_progress
    if _auto_ap_in_progress:
        return True

    # Do not interrupt while explicit user transition is running
    tr = get_transition_status()
    if tr.get("status") == "in_progress":
        return False

    status = get_wifi_status(cached=False)
    mode = (status.get("mode") or "").lower()
    ip = status.get("ip") or ""

    # If already in AP hotspot mode, or in client mode with valid IP, reset counter
    if mode in ("hotspot", "ap"):
        _offline_consecutive_checks = 0
        return False

    if mode == "client" and ip not in ("Disconnected", "10.42.0.1", "") and not ip.startswith("127."):
        _offline_consecutive_checks = 0
        return False

    # The network cannot connect to a WiFi / is offline
    _offline_consecutive_checks += 1
    print(f"[WiFiManager] Connection check: OFFLINE ({_offline_consecutive_checks}/{threshold_checks})")

    if _offline_consecutive_checks >= threshold_checks:
        _auto_ap_in_progress = True
        print("[WiFiManager] ⚠️ Network offline after checking connection! Auto-activating AP (Hotspot) mode...")
        cfg = load_network_config()

        def _worker():
            global _auto_ap_in_progress, _offline_consecutive_checks
            try:
                activate_hotspot(cfg.get("ap_ssid"), cfg.get("ap_password"))
            except Exception as e:
                print(f"[WiFiManager] Auto-AP activation error: {e}")
            finally:
                _auto_ap_in_progress = False
                _offline_consecutive_checks = 0

        t = threading.Thread(target=_worker, daemon=True, name="AutoAPFallbackWorker")
        t.start()
        return True

    return False


def start_wifi_watchdog(check_interval_s: float = 3.0):
    """Background watchdog thread monitoring network connectivity.
    
    The moment the network cannot connect to WiFi and status is offline after checking connection,
    it automatically falls back to Access Point (AP) mode.
    """
    global _wifi_watchdog_started
    if _wifi_watchdog_started:
        return
    _wifi_watchdog_started = True

    def _watchdog_loop():
        # Startup grace period to allow NetworkManager or init_startup_network to settle
        time.sleep(4.0)
        while True:
            try:
                time.sleep(check_interval_s)
                check_auto_ap_fallback(threshold_checks=2)
            except Exception:
                time.sleep(3.0)

    threading.Thread(target=_watchdog_loop, daemon=True, name="WiFiHealthWatchdog").start()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_startup_network()
    else:
        print(get_wifi_status())
