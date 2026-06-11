/*
 * 机械臂控制程序 v2（位姿驱动版）
 *
 * 相对你原版的主要变化：
 *  A. 新增标准二连杆【闭式逆解】 ik2link()，替代原来只控方向的
 *     computeAlphaPrime —— 现在能真正到达平面内目标点 (r, h)。
 *  B. 通道0 启用为【底座偏航 yaw】，与平面臂(alpha1/alpha2)组合后
 *     可覆盖 3D 位置。这正是"用上 translation"的关键。
 *  C. 新增笛卡尔指令  <C r;h;yaw;elbow>  —— 由 PC 端 grasp_to_arm.py
 *     从 6D 抓取位姿降维得到。臂端不碰四元数。
 *  D. 保留旧指令  <beta1;L4>  与原 computeAlphaPrime，向后兼容。
 *  E. 其余(状态机、millis 计时、范围校验、舵机方向/偏移)沿用你原版。
 *
 * 注意:四元数代表的完整三维朝向，这条臂的自由度不够实现;
 *      它只通过 PC 端被用于"接近向量退让 + 选肘向"。
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>

// ================= 机械臂参数（单位：mm） =================
const float L1 = 100.0;
const float L2 = 100.0;
float L4      = 50.0;   // 仅旧 beta1 指令用；动态可更新

// 当前关节角度（度）—— alpha2 为相对关节角（DH约定）
float alpha1     = 90.0;
float alpha2     = 90.0;
float beta1      = 30.0;
float base_yaw   = 0.0;   // 新增:底座偏航(度)，0 = 正前方

// ================= 舵机通道配置 =================
struct ServoConfig {
  int     channel;
  float   offset_deg;
  int8_t  direction;
};

// ----------------------------------------------------------------------
// 逻辑索引 -> 实际 PCA9685 通道映射（这里是唯一定义通道号的地方）
//   关节是【交叉】的: 关节1->通道3, 关节2->通道2
// ----------------------------------------------------------------------
enum {
  IDX_A1   = 0,   // 关节1 (alpha1)  -> 通道 3
  IDX_A2   = 1,   // 关节2 (alpha2)  -> 通道 2
  IDX_GRIP = 2,   // 夹爪            -> 通道 1  (本次不驱动)
  IDX_YAW  = 3,   // 底座左右旋转    -> 通道 4
};

const ServoConfig SERVO_CFG[] = {
  {3,  0.0f,  1},   // IDX_A1   关节1 -> 通道3
  {2,  0.0f,  1},   // IDX_A2   关节2 -> 通道2
  {1,  0.0f,  1},   // IDX_GRIP 夹爪  -> 通道1 (TODO: 本次留空)
  {4,  0.0f,  1},   // IDX_YAW  旋转  -> 通道4
};

// ================= PCA9685 =================
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
#define SERVOMIN 150
#define SERVOMAX 600

int angleToPulse(float angle_deg, int cfg_index) {
  const ServoConfig &cfg = SERVO_CFG[cfg_index];
  float phys = cfg.offset_deg + cfg.direction * (angle_deg - 90.0f) + 90.0f;
  phys = constrain(phys, 0.0f, 180.0f);
  return map((int)phys, 0, 180, SERVOMIN, SERVOMAX);
}

// 按逻辑索引驱动舵机（通道号从 SERVO_CFG 取，杜绝通道/下标混用）
void driveServo(int idx, float angle_deg) {
  pwm.setPWM(SERVO_CFG[idx].channel, 0, angleToPulse(angle_deg, idx));
}

// ================= 正运动学（DH 相对关节角约定）=================
void forwardKinematics(float a1_deg, float a2_deg, float &x, float &y) {
  float a1   = radians(a1_deg);
  float a2   = radians(a2_deg);
  float abs2 = a1 + a2;
  x = L1 * cos(a1) + L2 * cos(abs2);
  y = L1 * sin(a1) + L2 * sin(abs2);
}

// ================= 雅可比矩阵 =================
void computeJacobian(float a1_deg, float a2_deg, float J[2][2]) {
  float a1   = radians(a1_deg);
  float a2   = radians(a2_deg);
  float abs2 = a1 + a2;
  J[0][0] = -L1*sin(a1) - L2*sin(abs2);
  J[1][0] =  L1*cos(a1) + L2*cos(abs2);
  J[0][1] = -L2*sin(abs2);
  J[1][1] =  L2*cos(abs2);
}

// ================= 伪逆求关节增量 =================
bool computeDeltaTheta(float J[2][2], float e[2], float gain, float dq[2]) {
  float JTJ[2][2];
  JTJ[0][0] = J[0][0]*J[0][0] + J[1][0]*J[1][0];
  JTJ[0][1] = J[0][0]*J[0][1] + J[1][0]*J[1][1];
  JTJ[1][0] = JTJ[0][1];
  JTJ[1][1] = J[0][1]*J[0][1] + J[1][1]*J[1][1];

  float det = JTJ[0][0]*JTJ[1][1] - JTJ[0][1]*JTJ[1][0];
  if (fabsf(det) < 1e-6f) { dq[0] = dq[1] = 0.0f; return false; }

  float inv[2][2];
  inv[0][0] =  JTJ[1][1] / det;  inv[0][1] = -JTJ[0][1] / det;
  inv[1][0] = -JTJ[1][0] / det;  inv[1][1] =  JTJ[0][0] / det;

  float JTe[2] = { J[0][0]*e[0] + J[1][0]*e[1],
                   J[0][1]*e[0] + J[1][1]*e[1] };
  dq[0] = gain * (inv[0][0]*JTe[0] + inv[0][1]*JTe[1]);
  dq[1] = gain * (inv[1][0]*JTe[0] + inv[1][1]*JTe[1]);
  return true;
}

// ================= 新增:标准二连杆闭式逆解 =================
/*
 * 给定平面目标 (px, py) 与肘向 elbow(+1 up / -1 down)，
 * 解出 alpha1(绝对), alpha2(相对)。返回 false 表示超出臂展。
 *
 *   c2 = (px^2+py^2 - L1^2 - L2^2) / (2 L1 L2)
 *   a2 = atan2( ±sqrt(1-c2^2), c2 )         // 相对关节角，±选肘向
 *   a1 = atan2(py,px) - atan2(L2 sin a2, L1 + L2 cos a2)
 */
bool ik2link(float px, float py, int elbow, float &a1_deg, float &a2_deg) {
  float r2 = px*px + py*py;
  float c2 = (r2 - L1*L1 - L2*L2) / (2.0f * L1 * L2);
  if (c2 < -1.0f || c2 > 1.0f) return false;      // 不可达
  float s2 = sqrtf(1.0f - c2*c2);
  if (elbow < 0) s2 = -s2;                          // elbow down
  float a2 = atan2f(s2, c2);
  float a1 = atan2f(py, px) - atan2f(L2*sinf(a2), L1 + L2*cosf(a2));
  a1_deg = degrees(a1);
  a2_deg = degrees(a2);
  return true;
}

// ================= 旧版方向控制(保留兼容) =================
bool computeAlphaPrime(float L1_, float L2_, float L4_,
                       float a1_deg, float a2_deg, float beta1_deg,
                       float &a1p_deg, float &a2p_deg) {
  float a2 = radians(a2_deg);
  float b1 = radians(beta1_deg);
  float L3 = sqrtf(L1_*L1_ + L2_*L2_ + 2.0f*L1_*L2_*cosf(a2));
  float cos_beta2 = (L2_*L2_ + L3*L3 - L1_*L1_) / (2.0f*L2_*L3);
  cos_beta2 = constrain(cos_beta2, -1.0f, 1.0f);
  float beta2 = acosf(cos_beta2);
  float beta3 = (float)M_PI - b1 - beta2;
  float DeltaL3 = sqrtf(L3*L3 + L4_*L4_ - 2.0f*L3*L4_*cosf(beta3));
  if (DeltaL3 < 1e-6f) return false;
  float cos_a2p = (L1_*L1_ + L2_*L2_ - DeltaL3*DeltaL3) / (2.0f*L1_*L2_);
  cos_a2p = constrain(cos_a2p, -1.0f, 1.0f);
  float a2p = (a2 >= 0.0f) ? acosf(cos_a2p) : -acosf(cos_a2p);
  float L3p = sqrtf(L1_*L1_ + L2_*L2_ + 2.0f*L1_*L2_*cosf(a2p));
  if (L3p < 1e-6f) return false;
  float cos_g3 = (L1_*L1_ + L3p*L3p - L2_*L2_) / (2.0f*L1_*L3p);
  cos_g3 = constrain(cos_g3, -1.0f, 1.0f);
  float gamma3 = acosf(cos_g3);
  float cos_g2 = (L3p*L3p + DeltaL3*DeltaL3 - L4_*L4_) / (2.0f*L3p*DeltaL3);
  cos_g2 = constrain(cos_g2, -1.0f, 1.0f);
  float gamma2 = acosf(cos_g2);
  float a1p = b1 - gamma2 - gamma3;
  a1p_deg = degrees(a1p);
  a2p_deg = degrees(a2p);
  return true;
}

// ================= 底座偏航：立即置位(可按需改为缓动) =================
void setBaseYaw(float yaw_deg) {
  base_yaw = constrain(yaw_deg, -90.0f, 90.0f);
  // 把 [-90,90] 偏航映射到以 90° 为中心的逻辑角
  driveServo(IDX_YAW, 90.0f + base_yaw);    // 通道4
}

// ================= 轨迹状态机 =================
const int   NUM_WAYPOINTS        = 50;
const float ALPHA_GAIN           = 0.8f;
const float MAX_JOINT_SPEED_DEG  = 30.0f;
const unsigned long CTRL_PERIOD_MS = 50;

static float wp_x[NUM_WAYPOINTS];
static float wp_y[NUM_WAYPOINTS];

enum TrajState { IDLE, RUNNING };
TrajState trajState = IDLE;

int   traj_step      = 0;
float traj_a1        = 0.0f;
float traj_a2        = 0.0f;
float traj_target_a1 = 0.0f;
float traj_target_a2 = 0.0f;
unsigned long traj_last_ms = 0;

void startTrajectory(float target_a1, float target_a2) {
  float x_s, y_s, x_t, y_t;
  forwardKinematics(alpha1, alpha2, x_s, y_s);
  forwardKinematics(target_a1, target_a2, x_t, y_t);
  for (int i = 0; i < NUM_WAYPOINTS; i++) {
    float t = (float)i / (NUM_WAYPOINTS - 1);
    wp_x[i] = x_s + t * (x_t - x_s);
    wp_y[i] = y_s + t * (y_t - y_s);
  }
  traj_a1        = alpha1;
  traj_a2        = alpha2;
  traj_target_a1 = target_a1;
  traj_target_a2 = target_a2;
  traj_step      = 0;
  traj_last_ms   = millis();
  trajState      = RUNNING;
  Serial.print("Traj start -> ("); Serial.print(target_a1);
  Serial.print(", "); Serial.print(target_a2); Serial.println(")");
}

void updateTrajectory() {
  if (trajState != RUNNING) return;
  unsigned long now = millis();
  if (now - traj_last_ms < CTRL_PERIOD_MS) return;
  traj_last_ms = now;

  if (traj_step >= NUM_WAYPOINTS) {
    alpha1 = traj_a1; alpha2 = traj_a2; trajState = IDLE;
    Serial.print("Traj done. a1="); Serial.print(alpha1);
    Serial.print(" a2=");           Serial.println(alpha2);
    return;
  }

  float x_cur, y_cur;
  forwardKinematics(traj_a1, traj_a2, x_cur, y_cur);
  float e[2] = { wp_x[traj_step] - x_cur, wp_y[traj_step] - y_cur };

  if (fabsf(e[0]) < 0.5f && fabsf(e[1]) < 0.5f) { traj_step++; return; }

  float J[2][2];
  computeJacobian(traj_a1, traj_a2, J);
  float dq[2];
  if (computeDeltaTheta(J, e, ALPHA_GAIN, dq)) {
    float maxStep = MAX_JOINT_SPEED_DEG * (CTRL_PERIOD_MS / 1000.0f);
    dq[0] = constrain(dq[0], -maxStep, maxStep);
    dq[1] = constrain(dq[1], -maxStep, maxStep);
    traj_a1 = constrain(traj_a1 + dq[0], 0.0f, 180.0f);
    traj_a2 = constrain(traj_a2 + dq[1], 0.0f, 180.0f);
  }
  driveServo(IDX_A1, traj_a1);   // 关节1 -> 通道3
  driveServo(IDX_A2, traj_a2);   // 关节2 -> 通道2
  traj_step++;
}

// ================= 串口解析 =================
String inputBuf = "";
bool   dataReady = false;

// 新增:处理笛卡尔指令 <C r;h;yaw;elbow>
void handleCartesian(String body) {
  // body 形如 "r;h;yaw;elbow"
  float vals[4]; int idx = 0; int start = 0;
  for (int i = 0; i <= body.length() && idx < 4; i++) {
    if (i == body.length() || body.charAt(i) == ';') {
      vals[idx++] = body.substring(start, i).toFloat();
      start = i + 1;
    }
  }
  if (idx < 4) { Serial.println("ERR: <C r;h;yaw;elbow> needs 4 fields"); return; }

  float r = vals[0], h = vals[1], yaw = vals[2];
  int elbow = (vals[3] >= 0) ? 1 : -1;

  if (yaw < -90.0f || yaw > 90.0f) { Serial.println("ERR: yaw range"); return; }
  if (trajState == RUNNING) { Serial.println("BUSY: ignored"); return; }

  float ta1, ta2;
  if (!ik2link(r, h, elbow, ta1, ta2)) {
    Serial.println("ERR: target unreachable"); return;
  }
  // 角度落到舵机量程外说明姿态不可行(需标定 offset 或换肘向)
  if (ta1 < 0 || ta1 > 180 || ta2 < -180 || ta2 > 180) {
    Serial.println("ERR: IK out of servo range"); return;
  }
  ta1 = constrain(ta1, 0.0f, 180.0f);
  ta2 = constrain(ta2, 0.0f, 180.0f);

  setBaseYaw(yaw);          // 先转底座面向物体
  startTrajectory(ta1, ta2); // 再驱动平面臂
}

// 旧版:处理 <beta1;L4>
// Gripper command: <G angle>, e.g. <G 60> open, <G 120> close
void handleGripper(String body) {
  body.trim();
  if (trajState == RUNNING) { Serial.println("BUSY: ignored"); return; }
  float angle = body.toFloat();
  if (body.length() == 0) angle = GRIP_OPEN_DEG;
  if (angle < 0.0f || angle > 180.0f) { Serial.println("ERR: gripper angle range"); return; }
  driveServo(IDX_GRIP, angle);
  Serial.print("Gripper set. angle="); Serial.println(angle);
}

void handleBeta(String content) {
  int sep = content.indexOf(';');
  if (sep == -1) { Serial.println("ERR: <beta1;L4>"); return; }
  float nB = content.substring(0, sep).toFloat();
  float nL = content.substring(sep + 1).toFloat();
  if (nB < -180 || nB > 180) { Serial.println("ERR: beta1 range"); return; }
  if (nL < 0   || nL > 500)  { Serial.println("ERR: L4 range"); return; }
  beta1 = nB; L4 = nL;
  if (trajState == RUNNING) { Serial.println("BUSY: ignored"); return; }
  float ta1, ta2;
  if (!computeAlphaPrime(L1, L2, L4, alpha1, alpha2, beta1, ta1, ta2)) {
    Serial.println("ERR: IK degenerate"); return;
  }
  ta1 = constrain(ta1, 0.0f, 180.0f);
  ta2 = constrain(ta2, 0.0f, 180.0f);
  startTrajectory(ta1, ta2);
}

void processSerial() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n')      dataReady = true;
    else if (ch != '\r') inputBuf += ch;
  }
  if (!dataReady) return;
  inputBuf.trim();

  if (inputBuf.startsWith("<") && inputBuf.endsWith(">")) {
    String content = inputBuf.substring(1, inputBuf.length() - 1);
    content.trim();
    if (content.startsWith("C")) {
      // Cartesian: <C r;h;yaw;elbow>
      String body = content.substring(1); body.trim();
      handleCartesian(body);
    } else if (content.startsWith("G")) {
      // Gripper: <G angle>
      String body = content.substring(1); body.trim();
      handleGripper(body);
    } else {
      // Legacy: <beta1;L4>
      handleBeta(content);
    }
  } else {
    Serial.println("ERR: malformed packet");
  }
  inputBuf = ""; dataReady = false;
}

// ================= Setup / Loop =================
void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  setBaseYaw(0.0f);                 // 底座旋转 -> 通道4，回正
  driveServo(IDX_A1, alpha1);       // 关节1   -> 通道3
  driveServo(IDX_A2, alpha2);       // 关节2   -> 通道2
  driveServo(IDX_GRIP, GRIP_OPEN_DEG);  // gripper open at boot

  Serial.println("Ready.");
  Serial.println("  位姿驱动: <C r;h;yaw;elbow>  e.g. <C 150;20;15;1>");
  Serial.println("  旧版兼容: <beta1;L4>          e.g. <45.0;50.0>");
}

void loop() {
  processSerial();
  updateTrajectory();
}
