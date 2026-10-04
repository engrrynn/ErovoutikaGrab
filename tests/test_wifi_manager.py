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
