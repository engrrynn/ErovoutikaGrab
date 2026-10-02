#!/usr/bin/env python3
"""Launcher for the ErovoutikaGrab Robot LCD View-Only Fullscreen HUD.

Launches native Chromium in kiosk mode pointing to the unified robot server:
   http://localhost:5001/?device=lcd

Features:
- Ensures scripts/web_interface.py is active (starts it in background if not already running)
- Automatically binds to Wayland/X11 display environment (wayland-0 / DISPLAY=:0)
- Configures Chromium with kiosk flags to eliminate borders, search bars, and cursor clutter
"""

import os
import signal
import ssl
import subprocess
import sys
import time
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PORT = 5001
TARGET_URL = f"https://localhost:{PORT}/?device=lcd"


def setup_display_env():
    """Configures environment variables for Wayland/X11 graphical display on Raspberry Pi."""
    if not os.environ.get("XDG_RUNTIME_DIR") and os.path.exists("/run/user/1000"):
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
    if not os.environ.get("WAYLAND_DISPLAY") and os.path.exists("/run/user/1000/wayland-0"):
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
    if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"


import argparse


def is_server_running(port=PORT):
    ctx = ssl._create_unverified_context()
    for proto in ("https", "http"):
        try:
            url = f"{proto}://127.0.0.1:{port}/api/telemetry"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=1.5, context=ctx) as res:
                return res.status == 200
        except Exception:
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description="ErovoutikaGrab Robot LCD Fullscreen HUD Launcher")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    parser.add_argument("--mock", action="store_true", help="Launch with simulated hardware")
    args, _ = parser.parse_known_args()

    port = args.port
    target_url = f"https://localhost:{port}/?device=lcd"

    setup_display_env()

    print("\n=======================================================")
    print("   ErovoutikaGrab Robot LCD Fullscreen HUD Launcher    ")
    print("=======================================================\n")

    server_process = None
    if not is_server_running(port):
        print(f"[Setup] Web server not detected on port {port}. Launching web_interface.py...")
        web_script = os.path.join(BASE_DIR, "scripts/web_interface.py")
        cmd = [sys.executable, web_script, "--port", str(port)]
        if args.mock:
            cmd.append("--mock")
        server_process = subprocess.Popen(cmd, cwd=BASE_DIR)

        # Wait for server to become responsive
        max_wait = 15.0
        start_t = time.time()
        while time.time() - start_t < max_wait:
            if is_server_running(port):
                print(f"[Ready] Web server is live on port {port}.\n")
                break
            time.sleep(0.5)
        else:
            print("[Warning] Timed out waiting for server, attempting to launch browser anyway...")
    else:
        print(f"[Setup] Existing robot server detected on port {port}.\n")

    # Find browser binary
    # Enforce single launcher instance
    import fcntl
    try:
        lock_file = open("/tmp/erovoutikagrab_hud.lock", "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("[Launcher] Another instance of HUD launcher is already active. Exiting.")
        sys.exit(0)

    # Clean up any lingering kiosk processes
    subprocess.run(["pkill", "-f", "chromium.*device=lcd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    browser_bin = None
    for candidate in ["chromium", "chromium-browser", "firefox"]:
        path = subprocess.run(["which", candidate], stdout=subprocess.PIPE, text=True).stdout.strip()
        if path:
            browser_bin = path
            break

    if not browser_bin:
        print("[Error] No suitable web browser (chromium/firefox) found in system PATH.")
        if server_process:
            server_process.terminate()
        sys.exit(1)

    print(f"[Display] Launching {browser_bin} in Fullscreen Kiosk HUD mode...")
    print(f"Target URL: {target_url}")

    kiosk_cmd = [
        browser_bin,
        "--kiosk",
        f"--app={target_url}",
        "--incognito",
        "--user-data-dir=/tmp/chromium-kiosk",
        "--password-store=basic",
        "--use-mock-keychain",
        "--no-first-run",
        "--no-default-browser-check",
        "--noerrdialogs",
        "--disable-infobars",
        "--check-for-update-interval=31536000",
        "--overscroll-history-navigation=0",
        "--disable-pinch",
        "--ignore-certificate-errors",
        "--allow-insecure-localhost",
        target_url
    ]

    try:
        browser_proc = subprocess.Popen(kiosk_cmd)
        browser_proc.wait()
    except KeyboardInterrupt:
        print("\n[Shutdown] Caught user interrupt.")
    finally:
        if server_process:
            print("[Shutdown] Stopping background robot server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except Exception:
                server_process.kill()
        print("[Shutdown] HUD Launcher finished.")


if __name__ == "__main__":
    main()
