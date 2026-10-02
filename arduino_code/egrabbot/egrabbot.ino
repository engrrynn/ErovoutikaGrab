/**
 * ErovoutikaGrab - Modern Non-Blocking Firmware for Arduino Nano
 * Controlled via Bluetooth (HC-05 / EROBOTA75) from Raspberry Pi 4 / 5
 * 
 * Features:
 * - 9600 Baud rate (native HC-05 hardware Bluetooth UART rate)
 * - Calibrated constants synchronized directly with config/robot_config.yaml
 * - Non-blocking millis() servo smoothing (no delay() freezes during motion)
 * - Staggered boot attachment to prevent battery brownout / voltage sag
 * - Anti-stiction timed micro-nudging (<NUDGE:DIR,MS,PWM>)
 * - Direct L/R motor PWM control (<DRIVE:L,R>)
 * - Direct servo target angles (<SERVO:S1,S2,S3>)
 * - Macro pick/place/center sequences (<MACRO:NAME>)
 * - Bidirectional telemetry & ACK reporting (<STATUS:...>)
 * - Watchdog failsafe: automatically stops motors if connection drops
 * - Backwards-compatible fallback with legacy single-char commands (F, B, L, R, S, U, D, Z, O, C, N)
 */

#include <Servo.h>

// ==============================================================================
// 1. Calibrated Hardware Configuration (From config/robot_config.yaml)
// ==============================================================================

// Calibrated Servo Default Centers (Startup Neutral Alignment)
const int DEFAULT_S1_CENTER = 93;   // S1 Shoulder Neutral
const int DEFAULT_S2_CENTER = 45;   // S2 Elbow Neutral (Safe: prevents strain/binding)
const int DEFAULT_S3_CENTER = 110;  // S3 Gripper Neutral

// Calibrated Arm UP / STOW Posture (Retracted Travel Posture)
const int S1_UP = 93;
const int S2_UP = 45;

// Calibrated Arm DOWN / REACH Posture (Ground Contact Pick Reach)
const int S1_DOWN = 93;
const int S2_DOWN = 0;

// Calibrated Gripper Jaws
const int S3_OPEN = 110;  // Gripper Jaws Full Release
const int S3_CLOSE = 70;  // Gripper Jaws Firm Clamp (avoids servo stall)

// Calibrated Motor Speeds
const int DEFAULT_MAX_SPD = 255;    // max_speed
const int DEFAULT_BASE_SPD = 225;   // base_speed
const int DEFAULT_TURN_SPD = 240;   // turn_speed
const int DEFAULT_NUDGE_PWM = 245;  // nudge_pwm
const int DEFAULT_NUDGE_MS = 250;   // nudge_default_ms

// ==============================================================================
// 2. Hardware Pin Definitions
// ==============================================================================

// Servo pin definitions
const int SERVO1_PIN = 9;   // Shoulder (Base Upright 1)
const int SERVO2_PIN = 10;  // Elbow    (Base Upright 2)
const int SERVO3_PIN = 11;  // Gripper

Servo servo1;
Servo servo2;
Servo servo3;

// Motor Controller H-Bridge Wiring
// Note: On the physical Erovoutika chassis, Channel A connects to the Right Motor
// and Channel B connects to the Left Motor. This is transposed dynamically in the
// host communication adapter via 'swap_left_right: true' in config/robot_config.yaml.
const int PWMA = 3, AIN1 = 4, AIN2 = 5;  // Channel A (Physical Right)
const int PWMB = 6, BIN1 = 7, BIN2 = 8;  // Channel B (Physical Left)
const int outputs[] = {PWMA, AIN1, AIN2, PWMB, BIN1, BIN2};

// Current actual and target angles (initialized to calibrated safe center positions)
int currentS1 = DEFAULT_S1_CENTER, targetS1 = DEFAULT_S1_CENTER;
int currentS2 = DEFAULT_S2_CENTER, targetS2 = DEFAULT_S2_CENTER;
int currentS3 = DEFAULT_S3_CENTER, targetS3 = DEFAULT_S3_CENTER;

// Servo movement speed control
unsigned long lastServoStepTime = 0;
const unsigned long SERVO_STEP_INTERVAL_MS = 20; // Step every 20ms for smooth motion

// Motor speeds
int currentLeftPWM = 0;
int currentRightPWM = 0;
int mspd = DEFAULT_MAX_SPD;
int bspd = DEFAULT_BASE_SPD;

// Micro-nudge state
bool nudgeActive = false;
unsigned long nudgeEndTime = 0;

// Watchdog timer (auto-stop motors if no valid command received within timeout)
unsigned long lastCommandTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 1500;
bool watchdogEnabled = true;

// Telemetry heartbeat interval
unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL_MS = 250;

// Macro execution state
enum MacroState {
  MACRO_IDLE,
  MACRO_ARM_DOWN,
  MACRO_ARM_UP,
  MACRO_ARM_DEFAULT,
  MACRO_GRIP_OPEN,
  MACRO_GRIP_CLOSE,
  MACRO_FULL_PICK
};
MacroState activeMacro = MACRO_IDLE;
int macroStep = 0;
unsigned long macroStepTime = 0;

// Serial buffer for framed protocol: <COMMAND:PARAM1,PARAM2,...>
const int MAX_BUF = 64;
char serialBuffer[MAX_BUF];
int bufIndex = 0;
bool receivingPacket = false;

// Function prototypes
void motor(int L, int R);
void stopMotors();
void updateServos();
void updateMacros();
void sendTelemetry();
void processPacket(const char* packet);
void processLegacyChar(char c);

void setup() {
  // HC-05 hardware module default is 9600 baud
  Serial.begin(9600);

  for (int pin : outputs) {
    pinMode(pin, OUTPUT);
  }
  stopMotors();

  // Staggered servo attachment to prevent power supply brownout
  // Initialized at calibrated neutral centers to prevent elbow joint strain
  servo1.attach(SERVO1_PIN);
  servo1.write(DEFAULT_S1_CENTER);
  delay(300);

  servo2.attach(SERVO2_PIN);
  servo2.write(DEFAULT_S2_CENTER);
  delay(300);

  servo3.attach(SERVO3_PIN);
  servo3.write(DEFAULT_S3_CENTER);
  delay(300);

  currentS1 = DEFAULT_S1_CENTER; targetS1 = DEFAULT_S1_CENTER;
  currentS2 = DEFAULT_S2_CENTER; targetS2 = DEFAULT_S2_CENTER;
  currentS3 = DEFAULT_S3_CENTER; targetS3 = DEFAULT_S3_CENTER;

  lastCommandTime = millis();
  Serial.println(F("<READY:ErovoutikaGrab_v2.0>"));
}

void loop() {
  unsigned long now = millis();

  // 1. Process incoming Serial data
  while (Serial.available() > 0) {
    char inChar = Serial.read();

    if (inChar == '<') {
      receivingPacket = true;
      bufIndex = 0;
    } else if (inChar == '>') {
      if (receivingPacket) {
        serialBuffer[bufIndex] = '\0';
        processPacket(serialBuffer);
        receivingPacket = false;
        lastCommandTime = now;
      }
    } else if (receivingPacket) {
      if (bufIndex < MAX_BUF - 1) {
        serialBuffer[bufIndex++] = inChar;
      } else {
        // Buffer overflow, drop packet
        receivingPacket = false;
        bufIndex = 0;
      }
    } else {
      // Legacy single-character command support
      if (inChar > 32) { // Non-whitespace
        processLegacyChar(inChar);
        lastCommandTime = now;
      }
    }
  }

  // 2. Handle micro-nudge expiration
  if (nudgeActive && now >= nudgeEndTime) {
    stopMotors();
    nudgeActive = false;
    Serial.println(F("<ACK:NUDGE_DONE>"));
  }

  // 3. Watchdog check
  if (watchdogEnabled && (currentLeftPWM != 0 || currentRightPWM != 0)) {
    if (now - lastCommandTime > WATCHDOG_TIMEOUT_MS) {
      stopMotors();
      Serial.println(F("<WARN:WATCHDOG_STOP>"));
    }
  }

  // 4. Update non-blocking servo motion
  if (now - lastServoStepTime >= SERVO_STEP_INTERVAL_MS) {
    lastServoStepTime = now;
    updateServos();
  }

  // 5. Update multi-step macros
  updateMacros();

  // 6. Periodic telemetry heartbeat
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;
    sendTelemetry();
  }
}

// Update servos progressively towards targets with smart directional kinematics:
// - Lowering/reaching (extending S1 / lowering S2): S1 moves first, then S2
// - Lifting/stowing (raising S2 / retracting S1): S2 moves first, then S1
// - Servo 3 (Gripper) moves independently and continuously so it can always move during arm down
// - Only calls servo.write() when an angle actually changes, preserving clean jitter-free PWM pulses
void updateServos() {
  int deltaS1 = targetS1 - currentS1;
  int deltaS2 = targetS2 - currentS2;
  int deltaS3 = targetS3 - currentS3;

  // If all servos are already at target positions, return immediately (avoids interrupt jitter)
  if (deltaS1 == 0 && deltaS2 == 0 && deltaS3 == 0) {
    return;
  }

  // S2 moves first when lifting away from ground
  bool s2First = (deltaS2 > 0 && deltaS1 <= 0) || (deltaS2 > 0 && deltaS1 < 0);

  if (s2First) {
    if (currentS2 < targetS2) { currentS2++; servo2.write(currentS2); }
    else if (currentS2 > targetS2) { currentS2--; servo2.write(currentS2); }

    if (currentS2 == targetS2) {
      if (currentS1 < targetS1) { currentS1++; servo1.write(currentS1); }
      else if (currentS1 > targetS1) { currentS1--; servo1.write(currentS1); }
    }
  } else {
    if (currentS1 < targetS1) { currentS1++; servo1.write(currentS1); }
    else if (currentS1 > targetS1) { currentS1--; servo1.write(currentS1); }

    if (currentS1 == targetS1) {
      if (currentS2 < targetS2) { currentS2++; servo2.write(currentS2); }
      else if (currentS2 > targetS2) { currentS2--; servo2.write(currentS2); }
    }
  }

  // Smooth gripper without odd-step oscillation (independent of S1/S2 motion)
  if (deltaS3 != 0) {
    if (abs(currentS3 - targetS3) <= 2) {
      currentS3 = targetS3;
    } else if (currentS3 < targetS3) {
      currentS3 += 2;
    } else {
      currentS3 -= 2;
    }
    currentS3 = constrain(currentS3, 10, 180);
    servo3.write(currentS3);
  }
}

// Drive motors with PWM values (-255 to 255)
void motor(int L, int R) {
  L = constrain(L, -255, 255);
  R = constrain(R, -255, 255);

  currentLeftPWM = L;
  currentRightPWM = R;

  digitalWrite(AIN1, L <= 0 ? HIGH : LOW);
  digitalWrite(AIN2, L > 0 ? HIGH : LOW);
  digitalWrite(BIN1, R <= 0 ? HIGH : LOW);
  digitalWrite(BIN2, R > 0 ? HIGH : LOW);

  analogWrite(PWMA, abs(L));
  analogWrite(PWMB, abs(R));
}

void stopMotors() {
  motor(0, 0);
}

// Process framed packet: e.g. "DRIVE:180,180", "NUDGE:F,80,200", "SERVO:45,60,100"
void processPacket(const char* packet) {
  char cmd[16] = {0};
  const char* colon = strchr(packet, ':');

  if (colon == NULL) {
    // Command without arguments, e.g. "STOP", "PING"
    strncpy(cmd, packet, sizeof(cmd) - 1);
  } else {
    int cmdLen = colon - packet;
    if (cmdLen >= (int)sizeof(cmd)) cmdLen = sizeof(cmd) - 1;
    strncpy(cmd, packet, cmdLen);
    cmd[cmdLen] = '\0';
  }

  if (strcmp(cmd, "PING") == 0) {
    Serial.println(F("<PONG>"));
  }
  else if (strcmp(cmd, "STOP") == 0) {
    stopMotors();
    nudgeActive = false;
    Serial.println(F("<ACK:STOP>"));
  }
  else if (strcmp(cmd, "DRIVE") == 0 && colon != NULL) {
    int L = 0, R = 0;
    if (sscanf(colon + 1, "%d,%d", &L, &R) == 2) {
      nudgeActive = false;
      motor(L, R);
      Serial.println(F("<ACK:DRIVE>"));
    }
  }
  else if (strcmp(cmd, "NUDGE") == 0 && colon != NULL) {
    char dir = 'F';
    int durationMs = DEFAULT_NUDGE_MS;
    int pwm = DEFAULT_NUDGE_PWM;
    if (sscanf(colon + 1, "%c,%d,%d", &dir, &durationMs, &pwm) >= 2) {
      durationMs = constrain(durationMs, 10, 1000);
      pwm = constrain(pwm, 100, 255);
      nudgeActive = true;
      nudgeEndTime = millis() + durationMs;

      switch (dir) {
        case 'F': motor(pwm, pwm); break;
        case 'B': motor(-pwm, -pwm); break;
        case 'L': motor(-pwm, pwm); break; // Pivot left
        case 'R': motor(pwm, -pwm); break; // Pivot right
        default: stopMotors(); nudgeActive = false; break;
      }
      Serial.println(F("<ACK:NUDGE_START>"));
    }
  }
  else if (strcmp(cmd, "SERVO") == 0 && colon != NULL) {
    int s1 = 0, s2 = 0, s3 = 0;
    if (sscanf(colon + 1, "%d,%d,%d", &s1, &s2, &s3) == 3) {
      targetS1 = constrain(s1, 0, 180);
      targetS2 = constrain(s2, 0, 180);
      targetS3 = constrain(s3, 10, 180);
      Serial.println(F("<ACK:SERVO>"));
    }
  }
  else if (strcmp(cmd, "MACRO") == 0 && colon != NULL) {
    const char* macroName = colon + 1;
    if (strcmp(macroName, "CENTER") == 0) {
      targetS1 = DEFAULT_S1_CENTER;
      targetS2 = DEFAULT_S2_CENTER;
      targetS3 = DEFAULT_S3_CENTER;
      Serial.println(F("<ACK:MACRO_CENTER>"));
    } else if (strcmp(macroName, "DOWN") == 0) {
      targetS1 = S1_DOWN;
      targetS2 = S2_DOWN;
      Serial.println(F("<ACK:MACRO_DOWN>"));
    } else if (strcmp(macroName, "UP") == 0 || strcmp(macroName, "STOW") == 0) {
      targetS1 = S1_UP;
      targetS2 = S2_UP;
      Serial.println(F("<ACK:MACRO_UP>"));
    } else if (strcmp(macroName, "OPEN") == 0) {
      targetS3 = S3_OPEN;
      Serial.println(F("<ACK:MACRO_OPEN>"));
    } else if (strcmp(macroName, "CLOSE") == 0) {
      targetS3 = S3_CLOSE;
      Serial.println(F("<ACK:MACRO_CLOSE>"));
    } else if (strcmp(macroName, "DEFAULT") == 0) {
      targetS1 = S1_UP;
      targetS2 = S2_UP;
      targetS3 = S3_OPEN;
      Serial.println(F("<ACK:MACRO_DEFAULT>"));
    } else if (strcmp(macroName, "PICK") == 0) {
      // Begin coordinated pick sequence
      activeMacro = MACRO_FULL_PICK;
      macroStep = 0;
      macroStepTime = millis();
      Serial.println(F("<ACK:MACRO_PICK_STARTED>"));
    }
  }
  else if (strcmp(cmd, "GET_STATUS") == 0) {
    sendTelemetry();
  }
}

// Coordinated pick sequence state machine without blocking delays
void updateMacros() {
  if (activeMacro == MACRO_FULL_PICK) {
    unsigned long elapsed = millis() - macroStepTime;
    switch (macroStep) {
      case 0: // Step 0: Open gripper wide
        targetS3 = S3_OPEN;
        macroStep = 1;
        macroStepTime = millis();
        break;
      case 1: // Step 1: Wait for gripper open, then lower arm
        if (elapsed > 400 && abs(currentS3 - S3_OPEN) < 5) {
          targetS1 = S1_DOWN;
          targetS2 = S2_DOWN;
          macroStep = 2;
          macroStepTime = millis();
        }
        break;
      case 2: // Step 2: Wait for arm down (S1 reached first, then S2), then close gripper
        if (elapsed > 1600 && abs(currentS1 - S1_DOWN) < 5 && abs(currentS2 - S2_DOWN) < 5) {
          targetS3 = S3_CLOSE;
          macroStep = 3;
          macroStepTime = millis();
        }
        break;
      case 3: // Step 3: Wait for gripper close, then raise arm
        if (elapsed > 800) {
          targetS1 = S1_UP;
          targetS2 = S2_UP;
          macroStep = 4;
          macroStepTime = millis();
        }
        break;
      case 4: // Step 4: Confirm arm raised (S1 reached first, then S2)
        if (elapsed > 1600 && abs(currentS1 - S1_UP) < 5 && abs(currentS2 - S2_UP) < 5) {
          activeMacro = MACRO_IDLE;
          Serial.println(F("<ACK:MACRO_PICK_COMPLETE>"));
        }
        break;
    }
  }
}

// Transmit telemetry: <STATUS:L_PWM,R_PWM,S1,S2,S3,MACRO_ACTIVE>
void sendTelemetry() {
  Serial.print(F("<STATUS:"));
  Serial.print(currentLeftPWM);
  Serial.print(',');
  Serial.print(currentRightPWM);
  Serial.print(',');
  Serial.print(currentS1);
  Serial.print(',');
  Serial.print(currentS2);
  Serial.print(',');
  Serial.print(currentS3);
  Serial.print(',');
  Serial.print(activeMacro != MACRO_IDLE ? 1 : 0);
  Serial.println(F(">"));
}

// Legacy single-character fallback
void processLegacyChar(char c) {
  switch (c) {
    case 'F': motor(mspd, mspd); break;
    case 'B': motor(-mspd, -mspd); break;
    case 'L': motor(-DEFAULT_TURN_SPD, DEFAULT_TURN_SPD); break;
    case 'R': motor(DEFAULT_TURN_SPD, -DEFAULT_TURN_SPD); break;
    case 'G': motor(bspd, mspd); break;
    case 'I': motor(mspd, bspd); break;
    case 'S': stopMotors(); break;
    case 'U': targetS1 = S1_UP; targetS2 = S2_UP; break;
    case 'D': targetS1 = S1_DOWN; targetS2 = S2_DOWN; break;
    case 'Z': targetS1 = S1_UP; targetS2 = S2_UP; break;
    case 'O': targetS3 = S3_OPEN; break;
    case 'C': targetS3 = S3_CLOSE; break;
    case 'N': targetS1 = DEFAULT_S1_CENTER; targetS2 = DEFAULT_S2_CENTER; targetS3 = DEFAULT_S3_CENTER; break; // Neutral/Center
  }
}
