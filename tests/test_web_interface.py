"""Unit tests for the local web dashboard and API endpoints."""

import json
import os
import ssl
import threading
import time
import urllib.request
import pytest

from scripts.web_interface import (
    WebHandler,
    Port80RedirectHandler,
    connect_comm,
    load_all_configs,
    SSL_CERT_PATH,
    SSL_KEY_PATH,
)
from src.adapters.camera.mock_camera import MockCameraAdapter
from http.server import ThreadingHTTPServer
import scripts.web_interface as web_mod

# Configure urllib to accept self-signed test certificates in unit tests
_ssl_ctx = ssl._create_unverified_context()
_orig_urlopen = urllib.request.urlopen


def _test_urlopen(url_or_req, *args, **kwargs):
    if "context" not in kwargs:
        kwargs["context"] = _ssl_ctx
    return _orig_urlopen(url_or_req, *args, **kwargs)


urllib.request.urlopen = _test_urlopen


@pytest.fixture(scope="module")
def web_server():
    # Force mock mode
    web_mod.load_all_configs()
    web_mod.camera = MockCameraAdapter(width=640, height=480)
    web_mod.camera.start()
    web_mod.connect_comm(force_mock=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), WebHandler)
    port = server.server_address[1]
    proto = "http"
    if os.path.exists(SSL_CERT_PATH) and os.path.exists(SSL_KEY_PATH):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=SSL_CERT_PATH, keyfile=SSL_KEY_PATH)
        server.socket = ssl_ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    yield f"{proto}://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()
    if web_mod.camera:
        web_mod.camera.stop()
    if web_mod.comm:
        web_mod.comm.disconnect()


def test_index_page(web_server):
    res = urllib.request.urlopen(f"{web_server}/")
    assert res.status == 200
    html = res.read().decode("utf-8")
    assert "egrabbot.local:5001" in html
    assert "ErovoutikaGrab Dashboard" in html


def test_telemetry_endpoint(web_server):
    res = urllib.request.urlopen(f"{web_server}/api/telemetry")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert "link_status" in data
    assert "s1" in data
    assert "s2" in data
    assert "s3" in data


def test_drive_commands(web_server):
    for direction in ["F", "B", "L", "R"]:
        req = urllib.request.Request(
            f"{web_server}/api/drive_dir",
            data=json.dumps({"dir": direction}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        res = urllib.request.urlopen(req)
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        assert data["status"] == "ok"

    # Test stop
    req = urllib.request.Request(
        f"{web_server}/api/stop",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200


def test_servo_movement(web_server):
    req = urllib.request.Request(
        f"{web_server}/api/servo",
        data=json.dumps({"s1": 95, "s2": 40, "s3": 105}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["s1"] == 95
    assert data["s2"] == 40
    assert data["s3"] == 105


def test_macros(web_server):
    for macro in ["stow", "center", "open", "close"]:
        req = urllib.request.Request(
            f"{web_server}/api/macro",
            data=json.dumps({"name": macro}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        res = urllib.request.urlopen(req)
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        assert data["macro"] == macro
        assert data["success"] is True


def test_sweet_spot_update(web_server):
    req = urllib.request.Request(
        f"{web_server}/api/sweet_spot",
        data=json.dumps({"center_x": 320, "center_y": 240, "box_width": 200, "box_height": 180}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["sweet_spot"]["center_x"] == 320
    assert data["sweet_spot"]["center_y"] == 240


def test_feed_destination_switch(web_server):
    # Switch to web
    req = urllib.request.Request(
        f"{web_server}/api/feed_destination",
        data=json.dumps({"destination": "web"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["feed_destination"] == "web"

    # Switch back to lcd
    req = urllib.request.Request(
        f"{web_server}/api/feed_destination",
        data=json.dumps({"destination": "lcd"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["feed_destination"] == "lcd"


def test_static_logo(web_server):
    res = urllib.request.urlopen(f"{web_server}/logo.webp")
    assert res.status == 200
    assert res.headers.get("Content-Type") == "image/webp"
    data = res.read()
    assert len(data) > 1000


def test_autonomous_controls(web_server):
    # Start auto pick
    req = urllib.request.Request(
        f"{web_server}/api/auto/start",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["status"] == "ok"


def test_showcase_trigger(web_server):
    req = urllib.request.Request(
        f"{web_server}/api/showcase",
        data=b"{}",
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert data["status"] == "ok"


def test_port80_redirect_to_https():
    Port80RedirectHandler.protocol = "https"
    Port80RedirectHandler.target_port = 5001

    redirect_srv = ThreadingHTTPServer(("127.0.0.1", 5088), Port80RedirectHandler)
    t = threading.Thread(target=redirect_srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_302(self, req, fp, code, msg, headers):
            return fp

    opener = urllib.request.build_opener(NoRedirect)
    res = opener.open("http://127.0.0.1:5088/some_endpoint")
    assert res.headers.get("Location") == "https://127.0.0.1:5001/some_endpoint"

    redirect_srv.shutdown()
    redirect_srv.server_close()


def test_yoloe_telemetry(web_server):
    res = urllib.request.urlopen(f"{web_server}/api/telemetry")
    assert res.status == 200
    data = json.loads(res.read().decode("utf-8"))
    assert "yoloe" in data
    y = data["yoloe"]
    assert "object" in y
    assert "pickable" in y
    assert "aligned" in y
    assert "dx" in y
    assert "dy" in y
    assert "confidence_threshold" in y
    assert "imgsz" in y
    assert "model_path" in y


def test_yoloe_config_endpoints(web_server):
    # Test GET /api/yoloe/config
    res = urllib.request.urlopen(f"{web_server}/api/yoloe/config")
    assert res.status == 200
    cfg = json.loads(res.read().decode("utf-8"))
    assert cfg["status"] == "ok"
    assert "model_path" in cfg
    assert "confidence_threshold" in cfg
    assert "imgsz" in cfg
    assert "target_classes" in cfg
    assert "available_models" in cfg
    assert len(cfg["available_models"]) >= 1

    # Test POST /api/yoloe/config (live update in memory, no persist)
    req = urllib.request.Request(
        f"{web_server}/api/yoloe/config",
        data=json.dumps({
            "confidence_threshold": 0.42,
            "imgsz": 320,
            "target_classes": "can, bottle, cup",
            "save_yaml": False
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    post_res = json.loads(res.read().decode("utf-8"))
    assert post_res["status"] == "ok"
    assert post_res["confidence_threshold"] == 0.42
    assert post_res["imgsz"] == 320
    assert "can, bottle, cup" in post_res["target_classes"]


def test_servo_calibration_endpoints(web_server):
    # Test GET /api/servo_cal
    res = urllib.request.urlopen(f"{web_server}/api/servo_cal")
    assert res.status == 200
    cal_res = json.loads(res.read().decode("utf-8"))
    assert cal_res["status"] == "ok"
    assert "servo_cal" in cal_res
    assert "servo_cal_mtime" in cal_res
    cal = cal_res["servo_cal"]
    for key in ["s1_stow", "s1_down", "s1_center", "s2_stow", "s2_down", "s2_center", "s3_open", "s3_close", "s3_center"]:
        assert key in cal

    # Test POST /api/save_servos
    req = urllib.request.Request(
        f"{web_server}/api/save_servos",
        data=json.dumps(cal).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    assert res.status == 200
    save_res = json.loads(res.read().decode("utf-8"))
    assert save_res["status"] == "ok"
    assert "servo_cal" in save_res
    assert "servo_cal_mtime" in save_res



