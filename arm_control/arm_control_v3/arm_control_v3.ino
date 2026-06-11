#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

const float L1 = 130.0f;
const float L2 = 200.0f;

const float A1_PHYS_MIN = 0.0f;
const float A1_PHYS_MAX = 90.0f;
const float A2_PHYS_MIN = 0.0f;
const float A2_PHYS_MAX = 120.0f;
const float A2_STRAIGHT_PHYS = 0.0f;

float alpha1 = 90.0f;
float alpha2 = 120.0f;
float base_yaw = 0.0f;

struct ServoConfig {
  int channel;
  float phys_lo;
  float logic_lo;
  float phys_hi;
  float logic_hi;
};

enum {
  IDX_A1 = 0,
  IDX_A2 = 1,
  IDX_GRIP = 2,
  IDX_YAW = 3,
};

const ServoConfig SERVO_CFG[] = {
  {2, 0.0f, 20.0f, 90.0f, 100.0f},
  {1, 0.0f, 0.0f, 120.0f, 120.0f},
  {4, 0.0f, 0.0f, 180.0f, 180.0f},
  {3, -90.0f, 0.0f, 90.0f, 180.0f},
};

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
#define SERVOMIN 150
#define SERVOMAX 600

float physToLogic(int idx, float phys_deg) {
  const ServoConfig &c = SERVO_CFG[idx];
  float span = c.phys_hi - c.phys_lo;
  if (fabsf(span) < 1e-6f) return c.logic_lo;
  float t = (phys_deg - c.phys_lo) / span;
  t = constrain(t, 0.0f, 1.0f);
  return c.logic_lo + t * (c.logic_hi - c.logic_lo);
}

int logicToPulse(float logic_deg) {
  logic_deg = constrain(logic_deg, 0.0f, 180.0f);
  return map((int)logic_deg, 0, 180, SERVOMIN, SERVOMAX);
}

void driveServo(int idx, float phys_angle_deg) {
  float logic = physToLogic(idx, phys_angle_deg);
  pwm.setPWM(SERVO_CFG[idx].channel, 0, logicToPulse(logic));
}

void driveCurrentPose() {
  driveServo(IDX_YAW, base_yaw);
  driveServo(IDX_A1, alpha1);
  driveServo(IDX_A2, alpha2);
}

void forwardKinematics(float pa1_deg, float pa2_deg, float &x, float &y) {
  float a1 = radians(pa1_deg);
  float a2 = radians(A2_STRAIGHT_PHYS - pa2_deg);
  float abs2 = a1 + a2;
  x = L1 * cos(a1) + L2 * cos(abs2);
  y = L1 * sin(a1) + L2 * sin(abs2);
}

bool ik2link(float px, float py, int elbow, float &a1_deg, float &a2_deg) {
  float r2 = px * px + py * py;
  float c2 = (r2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
  if (c2 < -1.0f || c2 > 1.0f) return false;

  float s2 = sqrtf(max(0.0f, 1.0f - c2 * c2));
  if (elbow < 0) s2 = -s2;

  float dh_a2 = atan2f(s2, c2);
  float dh_a1 = atan2f(py, px) - atan2f(L2 * s2, L1 + L2 * c2);

  a1_deg = degrees(dh_a1);
  a2_deg = A2_STRAIGHT_PHYS - degrees(dh_a2);
  return true;
}

const int NUM_WAYPOINTS = 50;
const unsigned long CTRL_PERIOD_MS = 50;

enum TrajState { IDLE, RUNNING };
TrajState trajState = IDLE;

int traj_step = 0;
float traj_start_a1 = 90.0f;
float traj_start_a2 = 0.0f;
float traj_target_a1 = 90.0f;
float traj_target_a2 = 0.0f;
unsigned long traj_last_ms = 0;

float smoothStep01(float t) {
  t = constrain(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

void startTrajectory(float target_a1, float target_a2) {
  target_a1 = constrain(target_a1, A1_PHYS_MIN, A1_PHYS_MAX);
  target_a2 = constrain(target_a2, A2_PHYS_MIN, A2_PHYS_MAX);

  traj_start_a1 = alpha1;
  traj_start_a2 = alpha2;
  traj_target_a1 = target_a1;
  traj_target_a2 = target_a2;
  traj_step = 0;
  traj_last_ms = millis();
  trajState = RUNNING;

  Serial.print("Traj start -> target=(");
  Serial.print(traj_target_a1);
  Serial.print(", ");
  Serial.print(traj_target_a2);
  Serial.print(") start=(");
  Serial.print(traj_start_a1);
  Serial.print(", ");
  Serial.print(traj_start_a2);
  Serial.println(") [phys deg]");
}

void updateTrajectory() {
  if (trajState != RUNNING) return;

  unsigned long now = millis();
  if (now - traj_last_ms < CTRL_PERIOD_MS) return;
  traj_last_ms = now;

  traj_step++;
  float t = smoothStep01((float)traj_step / (float)NUM_WAYPOINTS);
  alpha1 = traj_start_a1 + t * (traj_target_a1 - traj_start_a1);
  alpha2 = traj_start_a2 + t * (traj_target_a2 - traj_start_a2);
  alpha1 = constrain(alpha1, A1_PHYS_MIN, A1_PHYS_MAX);
  alpha2 = constrain(alpha2, A2_PHYS_MIN, A2_PHYS_MAX);
  driveCurrentPose();

  if (traj_step >= NUM_WAYPOINTS) {
    alpha1 = traj_target_a1;
    alpha2 = traj_target_a2;
    driveCurrentPose();
    trajState = IDLE;
    Serial.print("Traj done. a1=");
    Serial.print(alpha1);
    Serial.print(" a2=");
    Serial.println(alpha2);
  }
}

String inputBuf = "";
bool dataReady = false;

bool parse4Floats(String body, float vals[4]) {
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= body.length() && idx < 4; i++) {
    if (i == body.length() || body.charAt(i) == ';') {
      vals[idx++] = body.substring(start, i).toFloat();
      start = i + 1;
    }
  }
  return idx == 4;
}

bool parse3Floats(String body, float vals[3]) {
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= body.length() && idx < 3; i++) {
    if (i == body.length() || body.charAt(i) == ';') {
      vals[idx++] = body.substring(start, i).toFloat();
      start = i + 1;
    }
  }
  return idx == 3;
}

void handleCartesian(String body) {
  float vals[4];
  if (!parse4Floats(body, vals)) {
    Serial.println("ERR: <C r;h;yaw;elbow> needs 4 fields");
    return;
  }

  float r = vals[0];
  float h = vals[1];
  float yaw = vals[2];
  int elbow = (vals[3] >= 0) ? 1 : -1;

  if (yaw < -90.0f || yaw > 90.0f) {
    Serial.println("ERR: yaw range");
    return;
  }
  if (trajState == RUNNING) {
    Serial.println("BUSY: ignored");
    return;
  }

  float ta1, ta2;
  if (!ik2link(r, h, elbow, ta1, ta2)) {
    Serial.println("ERR: target unreachable");
    return;
  }
  if (ta1 < A1_PHYS_MIN || ta1 > A1_PHYS_MAX ||
      ta2 < A2_PHYS_MIN || ta2 > A2_PHYS_MAX) {
    Serial.print("ERR: IK out of joint range a1=");
    Serial.print(ta1);
    Serial.print(" a2=");
    Serial.println(ta2);
    return;
  }

  base_yaw = constrain(yaw, -90.0f, 90.0f);
  driveServo(IDX_YAW, base_yaw);
  startTrajectory(ta1, ta2);
}

void handleJoint(String body) {
  float vals[3];
  if (!parse3Floats(body, vals)) {
    Serial.println("ERR: <J angle1;angle2;yaw> needs 3 fields");
    return;
  }
  if (trajState == RUNNING) {
    Serial.println("BUSY: ignored");
    return;
  }

  alpha1 = constrain(vals[0], A1_PHYS_MIN, A1_PHYS_MAX);
  alpha2 = constrain(vals[1], A2_PHYS_MIN, A2_PHYS_MAX);
  base_yaw = constrain(vals[2], -90.0f, 90.0f);
  driveCurrentPose();

  Serial.print("Joint set. a1=");
  Serial.print(alpha1);
  Serial.print(" a2=");
  Serial.print(alpha2);
  Serial.print(" yaw=");
  Serial.println(base_yaw);
}

void processPacket(String packet) {
  packet.trim();
  if (packet.length() == 0) return;

  if (!(packet.startsWith("<") && packet.endsWith(">"))) {
    Serial.println("ERR: malformed packet");
    return;
  }

  String content = packet.substring(1, packet.length() - 1);
  content.trim();

  if (content == "PING") {
    Serial.println("PONG");
    return;
  }

  if (content.startsWith("C")) {
    String body = content.substring(1);
    body.trim();
    handleCartesian(body);
    return;
  }

  if (content.startsWith("J")) {
    String body = content.substring(1);
    body.trim();
    handleJoint(body);
    return;
  }

  Serial.println("ERR: unknown command");
}

void processSerial() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      dataReady = true;
      break;
    }
    inputBuf += ch;
    if (ch == '>') {
      dataReady = true;
      break;
    }
    if (inputBuf.length() > 96) {
      inputBuf = "";
      dataReady = false;
      Serial.println("ERR: packet overflow");
      return;
    }
  }

  if (!dataReady) return;
  String packet = inputBuf;
  inputBuf = "";
  dataReady = false;
  processPacket(packet);
}

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  driveCurrentPose();

  Serial.println("Ready. arm_control_v3 clean");
  Serial.println("Cartesian: <C r;h;yaw;elbow>");
  Serial.println("Joint test: <J angle1;angle2;yaw>");
  Serial.println("Ping: <PING>");
}

void loop() {
  processSerial();
  updateTrajectory();
}
