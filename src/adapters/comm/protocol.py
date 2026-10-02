"""Framed ASCII communication protocol encoder and parser for Arduino Nano."""

import re
from typing import Any, Dict, Optional, Tuple


class ProtocolError(Exception):
    """Protocol parsing or validation exception."""
    pass


class ArduinoProtocol:
    """Encodes outgoing commands and parses incoming responses."""

    @staticmethod
    def encode_command(cmd: str, *args) -> bytes:
        """Encodes command into framed ASCII format: <CMD:arg1,arg2,...>\\n"""
        if args:
            args_str = ",".join(str(a) for a in args)
            packet = f"<{cmd}:{args_str}>\n"
        else:
            packet = f"<{cmd}>\n"
        return packet.encode("ascii")

    @staticmethod
    def encode_drive(left_pwm: int, right_pwm: int) -> bytes:
        left_pwm = max(-255, min(255, int(left_pwm)))
        right_pwm = max(-255, min(255, int(right_pwm)))
        return ArduinoProtocol.encode_command("DRIVE", left_pwm, right_pwm)

    @staticmethod
    def encode_stop() -> bytes:
        return ArduinoProtocol.encode_command("STOP")

    @staticmethod
    def encode_nudge(direction: str, duration_ms: int = 70, pwm: int = 210) -> bytes:
        direction = direction.upper()
        if direction not in ("F", "B", "L", "R"):
            raise ValueError(f"Invalid nudge direction: {direction}")
        duration_ms = max(10, min(1000, int(duration_ms)))
        pwm = max(100, min(255, int(pwm)))
        return ArduinoProtocol.encode_command("NUDGE", direction, duration_ms, pwm)

    @staticmethod
    def encode_servos(s1: int, s2: int, s3: int) -> bytes:
        s1 = max(0, min(180, int(s1)))
        s2 = max(0, min(180, int(s2)))
        s3 = max(10, min(180, int(s3)))
        return ArduinoProtocol.encode_command("SERVO", s1, s2, s3)

    @staticmethod
    def encode_macro(name: str) -> bytes:
        name = name.upper()
        valid = ("PICK", "DOWN", "UP", "OPEN", "CLOSE", "DEFAULT", "CENTER")
        if name not in valid:
            raise ValueError(f"Unknown macro: {name}. Must be one of {valid}")
        return ArduinoProtocol.encode_command("MACRO", name)

    @staticmethod
    def encode_ping() -> bytes:
        return ArduinoProtocol.encode_command("PING")

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, Any]]:
        """Parses a received string into a structured packet dictionary.
        
        Supported incoming packets:
        - <STATUS:L,R,S1,S2,S3,MACRO_ACTIVE>
        - <ACK:CMD_NAME>
        - <PONG>
        - <READY:VERSION>
        - <WARN:MSG>
        """
        line = line.strip()
        if not (line.startswith("<") and line.endswith(">")):
            return None

        content = line[1:-1].strip()
        if ":" in content:
            header, payload = content.split(":", 1)
        else:
            header, payload = content, ""

        if header == "STATUS":
            parts = payload.split(",")
            if len(parts) >= 6:
                try:
                    return {
                        "type": "STATUS",
                        "left_pwm": int(parts[0]),
                        "right_pwm": int(parts[1]),
                        "s1": int(parts[2]),
                        "s2": int(parts[3]),
                        "s3": int(parts[4]),
                        "macro_active": bool(int(parts[5]))
                    }
                except ValueError:
                    return None
            return None
        elif header == "ACK":
            return {"type": "ACK", "command": payload}
        elif header == "PONG":
            return {"type": "PONG"}
        elif header == "READY":
            return {"type": "READY", "version": payload}
        elif header == "WARN":
            return {"type": "WARN", "message": payload}

        return {"type": header, "payload": payload}
