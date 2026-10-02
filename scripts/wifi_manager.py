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
from typing import Dict, List, Any, Optional
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_CONFIG_PATH = os.path.join(BASE_DIR, "config/network_config.yaml")


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


def get_wifi_status() -> Dict[str, Any]:
    """Inspects live wireless device status via nmcli."""
    res = {
        "mode": "disconnected",
        "ssid": "None",
        "ip": get_current_ip(),
        "signal": 0,
        "security": "None",
        "device": "wlan0",
        "startup_mode": load_network_config().get("startup_mode", "ap")
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
                    res["ssid"] = load_network_config().get("ap_ssid", "egrabbot-ap")
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


def connect_to_wifi(ssid: str, password: Optional[str] = None) -> Dict[str, Any]:
    """Seamlessly connects the robot to a specified WiFi network."""
    if not ssid or not ssid.strip():
        return {"status": "error", "message": "SSID cannot be empty."}

    ssid = ssid.strip()
    password = password.strip() if (password and password.strip()) else None

    # Step 1: If AP hotspot is currently active on wlan0, deactivate it cleanly
    try:
        chk_ap = run_cmd(["nmcli", "-t", "-f", "NAME,DEVICE", "con", "show", "--active"])
        for line in chk_ap.stdout.splitlines():
            if "egrabbot-hotspot" in line:
                run_cmd(["sudo", "nmcli", "con", "down", "egrabbot-hotspot"], timeout_s=6.0)
                break
    except Exception:
        pass

    saved = get_saved_wifi_connections()

    # Step 2: Try activating existing saved connection profile if available
    if ssid in saved:
        if password:
            # Update password with explicit key-mgmt wpa-psk
            run_cmd([
                "sudo", "nmcli", "con", "modify", ssid,
                "802-11-wireless-security.key-mgmt", "wpa-psk",
                "802-11-wireless-security.psk", password
            ], timeout_s=5.0)

        up_proc = run_cmd(["sudo", "nmcli", "con", "up", ssid], timeout_s=20.0)
        if up_proc.returncode == 0:
            cfg = load_network_config()
            cfg["last_connected_ssid"] = ssid
            save_network_config(cfg)
            new_ip = get_current_ip()
            return {
                "status": "ok",
                "message": f"Successfully connected to saved network '{ssid}'!",
                "ssid": ssid,
                "ip": new_ip
            }

    # Step 3: Try nmcli dev wifi connect
    cmd = ["sudo", "nmcli", "dev", "wifi", "connect", ssid]
    if password:
        cmd.extend(["password", password])

    try:
        proc = run_cmd(cmd, timeout_s=25.0)
        if proc.returncode == 0:
            cfg = load_network_config()
            cfg["last_connected_ssid"] = ssid
            save_network_config(cfg)
            new_ip = get_current_ip()
            return {
                "status": "ok",
                "message": f"Successfully connected to '{ssid}'!",
                "ssid": ssid,
                "ip": new_ip
            }
        
        err = proc.stderr.strip() or proc.stdout.strip()

        # Step 4: If dev wifi connect failed with 802-11-wireless-security error, explicitly add profile with full key-mgmt
        if password:
            run_cmd(["sudo", "nmcli", "con", "delete", ssid], timeout_s=4.0)
            add_cmd = [
                "sudo", "nmcli", "con", "add", "type", "wifi", "ifname", "wlan0",
                "con-name", ssid, "ssid", ssid,
                "802-11-wireless-security.key-mgmt", "wpa-psk",
                "802-11-wireless-security.psk", password
            ]
            add_proc = run_cmd(add_cmd, timeout_s=8.0)
            if add_proc.returncode == 0:
                up_proc = run_cmd(["sudo", "nmcli", "con", "up", ssid], timeout_s=20.0)
                if up_proc.returncode == 0:
                    cfg = load_network_config()
                    cfg["last_connected_ssid"] = ssid
                    save_network_config(cfg)
                    return {
                        "status": "ok",
                        "message": f"Successfully configured and connected to '{ssid}'!",
                        "ssid": ssid,
                        "ip": get_current_ip()
                    }
                err = up_proc.stderr.strip() or up_proc.stdout.strip()
            else:
                err = add_proc.stderr.strip() or add_proc.stdout.strip()

        return {"status": "error", "message": f"Failed to connect: {err}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Connection timed out. Check password or router range."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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


def init_startup_network():
    """Initializes wireless networking on startup according to configuration."""
    cfg = load_network_config()
    target_mode = cfg.get("startup_mode", "ap")
    print(f"[WiFiManager] Initializing network in '{target_mode}' mode...")

    if target_mode == "ap":
        res = activate_hotspot(cfg.get("ap_ssid"), cfg.get("ap_password"))
        print(f"[WiFiManager] Hotspot init result: {res}")
    else:
        last_ssid = cfg.get("last_connected_ssid")
        if last_ssid:
            print(f"[WiFiManager] Bringing up saved WiFi: {last_ssid}")
            run_cmd(["sudo", "nmcli", "con", "up", last_ssid], timeout_s=15.0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_startup_network()
    else:
        print(get_wifi_status())
