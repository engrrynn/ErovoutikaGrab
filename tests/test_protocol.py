"""Unit tests for Arduino framed ASCII protocol encoder and decoder."""

import pytest
from src.adapters.comm.protocol import ArduinoProtocol


def test_encode_drive():
    packet = ArduinoProtocol.encode_drive(180, -180)
    assert packet == b"<DRIVE:180,-180>\n"

    # Test clamping to [-255, 255]
    packet_clamped = ArduinoProtocol.encode_drive(300, -500)
    assert packet_clamped == b"<DRIVE:255,-255>\n"


def test_encode_stop():
    packet = ArduinoProtocol.encode_stop()
    assert packet == b"<STOP>\n"


def test_encode_nudge():
    packet = ArduinoProtocol.encode_nudge("F", duration_ms=75, pwm=190)
    assert packet == b"<NUDGE:F,75,190>\n"

    with pytest.raises(ValueError):
        ArduinoProtocol.encode_nudge("X", 50, 150)


def test_encode_servos():
    packet = ArduinoProtocol.encode_servos(50, 45, 120)
    assert packet == b"<SERVO:50,45,120>\n"

    # Test servo limits clamping (0 to 180)
    packet_clamped = ArduinoProtocol.encode_servos(-10, 90, 250)
    assert packet_clamped == b"<SERVO:0,90,180>\n"


def test_encode_macro():
    assert ArduinoProtocol.encode_macro("PICK") == b"<MACRO:PICK>\n"
    assert ArduinoProtocol.encode_macro("DEFAULT") == b"<MACRO:DEFAULT>\n"
    assert ArduinoProtocol.encode_macro("CENTER") == b"<MACRO:CENTER>\n"

    with pytest.raises(ValueError):
        ArduinoProtocol.encode_macro("INVALID_MACRO")


def test_encode_ping():
    assert ArduinoProtocol.encode_ping() == b"<PING>\n"


def test_parse_status_packet():
    raw = "<STATUS:180,180,25,65,160,0>\n"
    parsed = ArduinoProtocol.parse_line(raw)
    assert parsed is not None
    assert parsed["type"] == "STATUS"
    assert parsed["left_pwm"] == 180
    assert parsed["right_pwm"] == 180
    assert parsed["s1"] == 25
    assert parsed["s2"] == 65
    assert parsed["s3"] == 160
    assert parsed["macro_active"] is False


def test_parse_ack_packet():
    raw = "<ACK:DRIVE>\r\n"
    parsed = ArduinoProtocol.parse_line(raw)
    assert parsed == {"type": "ACK", "command": "DRIVE"}


def test_parse_pong():
    raw = "<PONG>"
    parsed = ArduinoProtocol.parse_line(raw)
    assert parsed == {"type": "PONG"}


def test_parse_invalid():
    assert ArduinoProtocol.parse_line("NOT_A_PACKET") is None
    assert ArduinoProtocol.parse_line("<INVALID") is None
