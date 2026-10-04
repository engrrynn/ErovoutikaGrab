"""Unit tests for wifi_manager transition and configuration subsystem."""

from unittest.mock import patch, MagicMock
import scripts.wifi_manager as wm


def test_transition_status():
    status = wm.get_transition_status()
    assert isinstance(status, dict)
    assert "status" in status


def test_empty_ssid():
    res = wm.connect_to_wifi("")
    assert res["status"] == "error"
    assert "empty" in res["message"].lower()

    ok = wm.configure_wifi_profile("")
    assert ok is False


def test_connect_to_wifi_from_ap_returns_switching_prompt():
    with patch("scripts.wifi_manager.get_active_wifi_connection", return_value="egrabbot-hotspot"), \
         patch("scripts.wifi_manager.configure_wifi_profile", return_value=True), \
         patch("scripts.wifi_manager._async_transition_worker") as mock_worker:
        
        res = wm.connect_to_wifi("HomeRouter_2.4G", "mySecretPass123")
        assert res["status"] == "switching"
        assert res["from_ap"] is True
        assert res["target_ssid"] == "HomeRouter_2.4G"
        assert "Erovoutika_Grab_Bot" in res["message"]
        assert "egrabbot.local:5001" in res["message"]


def test_connect_to_wifi_profile_config_failure():
    with patch("scripts.wifi_manager.get_active_wifi_connection", return_value="egrabbot-hotspot"), \
         patch("scripts.wifi_manager.configure_wifi_profile", return_value=False):
        
        res = wm.connect_to_wifi("HomeRouter_2.4G", "pass")
        assert res["status"] == "error"
        assert "Failed to save profile" in res["message"]


def test_init_startup_network_ap_mode():
    with patch("scripts.wifi_manager.load_network_config", return_value={"startup_mode": "ap", "ap_ssid": "Erovoutika_Grab_Bot", "ap_password": "pass"}), \
         patch("scripts.wifi_manager.activate_hotspot", return_value={"status": "ok", "mode": "hotspot"}) as mock_ap:
        res = wm.init_startup_network()
        mock_ap.assert_called_once_with("Erovoutika_Grab_Bot", "pass")
        assert res["status"] == "ok"


def test_init_startup_network_wifi_failsafe_fallback_to_ap():
    # If configured as wifi, but no connection succeeds, it MUST fall back to AP mode
    with patch("scripts.wifi_manager.load_network_config", return_value={"startup_mode": "wifi", "last_connected_ssid": "UnreachableRouter", "ap_ssid": "Erovoutika_Grab_Bot", "ap_password": "pass"}), \
         patch("scripts.wifi_manager.get_active_wifi_connection", return_value=None), \
         patch("scripts.wifi_manager.get_current_ip", return_value="Disconnected"), \
         patch("scripts.wifi_manager.run_cmd", return_value=MagicMock(returncode=1, stderr="Not found")), \
         patch("scripts.wifi_manager.get_saved_wifi_connections", return_value=[]), \
         patch("scripts.wifi_manager.activate_hotspot", return_value={"status": "ok", "mode": "hotspot"}) as mock_ap:
        
        res = wm.init_startup_network()
        mock_ap.assert_called_once_with("Erovoutika_Grab_Bot", "pass")
        assert res.get("fail_safe_triggered") is True
        assert res["status"] == "ok"


def test_get_wifi_status_caching():
    with patch("scripts.wifi_manager.run_cmd", return_value=MagicMock(stdout="egrabbot-hotspot:wifi:wlan0\n")):
        res1 = wm.get_wifi_status(cached=False)
        assert "mode" in res1
        res2 = wm.get_wifi_status(cached=True)
        assert res2["mode"] == res1["mode"]

