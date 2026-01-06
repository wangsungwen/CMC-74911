/*
 * ===================================================================
 * ESP8266 雷射追蹤控制器 (PID 升級版)
 *
 * 理論基礎:
 * 採用雙 PID 閉迴路反饋系統 (一個用於 Pan, 一個用於 Tilt)。
 * 1. AI 伺服器發布 "tracker/target" (猴子座標) 和 "tracker/spot" (雷射座標)。
 * 2. ESP8266 接收這兩個座標。
 * 3. PID 控制器計算兩者之間的 "像素誤差" (Setpoint = 0)。
 * - Input:    當前誤差 (target - spot)
 * - Setpoint: 期望誤差 (0)
 * - Output:   驅動馬達的 "修正速度"
 * 4. AccelStepper 函式庫根據 PID 輸出的速度來驅動馬達。
 * 5. 增加 "Timeout" 機制，若 1 秒內未收到新目標，則停止馬達並禁用驅動器。
 * ===================================================================
 */
 /*
 需要先在 Arduino IDE 的「程式庫管理員」中安裝以下函式庫：
 PubSubClient (by Nick O'Leary)
 AccelStepper (by Mike McCauley)
 ArduinoJson (by Benoit Blanchon)
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>
#include <ArduinoJson.h>
#include <PID_v1.h> // 導入 PID 函式庫

// --- 1. WiFi 與 MQTT 設定 ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "YOUR_MQTT_BROKER_IP";
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP8266_LaserTracker_PID";

// --- 2. 步進馬達腳位設定 ---
// 旋迴 (Pan) 馬達 (X 軸)
#define PAN_STEP_PIN D1 // GPIO 5
#define PAN_DIR_PIN  D2 // GPIO 4

// 俯仰 (Tilt) 馬達 (Y 軸)
#define TILT_STEP_PIN D5 // GPIO 14
#define TILT_DIR_PIN  D6 // GPIO 12

// 步進馬達驅動器致能 (Enable) 腳位 (低電位致能)
// 將兩個驅動器的 EN 腳位連到同一個 ESP8266 腳位
#define STEPPER_ENABLE_PIN D8 // GPIO 15

AccelStepper panStepper(AccelStepper::DRIVER, PAN_STEP_PIN, PAN_DIR_PIN);
AccelStepper tiltStepper(AccelStepper::DRIVER, TILT_STEP_PIN, TILT_DIR_PIN);

// --- 3. 全域座標與狀態 ---
volatile float targetMonkeyX = 0.0;
volatile float targetMonkeyY = 0.0;
volatile float laserSpotX = 0.0;
volatile float laserSpotY = 0.0;

volatile bool newMonkeyData = false;
volatile bool newSpotData = false;

// Timeout 計時器
unsigned long lastDataTimestamp = 0;
const long dataTimeoutMs = 1000; // 1 秒

// --- 4. PID 控制器設定 ---
// PID 變數
double panError, panOutput, panSetpoint;
double tiltError, tiltOutput, tiltSetpoint;

/*
 * !! 關鍵調校參數 (Kp, Ki, Kd) !!
 * 這些值 *必須* 透過實際測試來調校 (Tuning)。
 * Kp (比例): 主要驅動力。太大會震盪，太小反應慢。
 * Ki (積分): 消除穩態誤差。太大會導致積分飽和 (windup)，反應變慢。
 * Kd (微分): 抑制震盪，增加穩定性。太大會對雜訊過度敏感。
 *
 * 建議調校順序:
 * 1. 先設 Ki = 0, Kd = 0。
 * 2. 逐漸增加 Kp，直到系統開始在目標附近 "穩定震盪"。
 * 3. 增加 Kd，直到震盪被抑制。
 * 4. 如果有穩態誤差 (一直差一點點)，再緩慢增加 Ki。
 */

// Pan (X 軸) PID 參數
double Kp_pan = 2.0;
double Ki_pan = 0.1;
double Kd_pan = 0.5;

// Tilt (Y 軸) PID 參數
double Kp_tilt = 2.0;  // Y 軸的 Kp, Ki, Kd 可能需要設為負值，
double Ki_tilt = 0.1;  // 取決於您的馬達安裝方向和座標系定義
double Kd_tilt = 0.5;  // (例如攝像頭 Y 軸向下為正)

// 建立 PID 控制器實例
// PID::SetMode(AUTOMATIC)     - PID 自動運行
// PID::SetSampleTime(10)      - 每 10ms 計算一次
// PID::SetOutputLimits(-500, 500) - 限制 PID 輸出值 (馬達最大速度)
PID panPID(&panError, &panOutput, &panSetpoint, Kp_pan, Ki_pan, Kd_pan, DIRECT);
PID tiltPID(&tiltError, &tiltOutput, &tiltSetpoint, Kp_tilt, Ki_tilt, Kd_tilt, DIRECT);

// --- 5. 網路與 MQTT ---
WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("正在連線到 ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi 已連線, IP: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  // 將 payload 轉換為字串
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';
  Serial.printf("收到訊息 [%s]: %s\n", topic, message);

  StaticJsonDocument<128> doc;
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.printf("JSON 解析失敗: %s\n", error.c_str());
    return;
  }

  // 更新座標
  if (strcmp(topic, "tracker/target") == 0) {
    targetMonkeyX = doc["x"];
    targetMonkeyY = doc["y"];
    newMonkeyData = true;
  } 
  else if (strcmp(topic, "tracker/spot") == 0) {
    laserSpotX = doc["x"];
    laserSpotY = doc["y"];
    newSpotData = true;
  }

  // 只要收到任何一筆資料，就重置 timeout 計時器
  lastDataTimestamp = millis();
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("嘗試 MQTT 連線中...");
    if (client.connect(mqtt_client_id)) {
      Serial.println("已連線");
      client.subscribe("tracker/target");
      client.subscribe("tracker/spot");
    } else {
      Serial.printf("失敗, rc=%d, 5 秒後重試\n", client.state());
      delay(5000);
    }
  }
}

// --- 6. Arduino Setup ---
void setup() {
  Serial.begin(115200);

  // 設定馬達致能腳位
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);
  digitalWrite(STEPPER_ENABLE_PIN, HIGH); // HIGH = 禁用驅動器 (省電/降溫)

  // 設定馬達參數
  panStepper.setMaxSpeed(1000.0);
  panStepper.setAcceleration(500.0);
  
  tiltStepper.setMaxSpeed(1000.0);
  tiltStepper.setAcceleration(500.0);

  // 初始化 WiFi 和 MQTT
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // 初始化 PID
  panSetpoint = 0.0;  // 我們的目標是讓誤差 (Error) 保持在 0
  tiltSetpoint = 0.0; // 我們的目標是讓誤差 (Error) 保持在 0

  // 設定 PID 輸出限制 (對應 AccelStepper 的 .setSpeed() 範圍)
  panPID.SetOutputLimits(-800, 800);   // 限制 Pan 馬達最大速度
  tiltPID.SetOutputLimits(-800, 800);  // 限制 Tilt 馬達最大速度
  
  // 設定 PID 運作模式
  panPID.SetMode(AUTOMATIC);
  tiltPID.SetMode(AUTOMATIC);

  lastDataTimestamp = millis();
}

// --- 7. Arduino Loop ---
void loop() {
  // 保持網路連線
  if (WiFi.status() != WL_CONNECTED) {
    setup_wifi();
  }
  if (!client.connected()) {
    reconnect();
  }
  client.loop(); // 處理 MQTT 訊息 (會觸發 callback)

  unsigned long now = millis();

  // 檢查是否 Timeout
  if (now - lastDataTimestamp > dataTimeoutMs) {
    // --- 目標丟失 / Timeout ---
    digitalWrite(STEPPER_ENABLE_PIN, HIGH); // 禁用馬達
    panStepper.setSpeed(0); // 停止馬達
    tiltStepper.setSpeed(0);
    
    // 重置積分項，防止積分暴衝 (Integral Windup)
    panPID.SetMode(MANUAL); // 關閉 PID
    panError = 0; panOutput = 0; // 手動清零
    panPID.SetMode(AUTOMATIC); // 重新開啟
    
    tiltPID.SetMode(MANUAL);
    tiltError = 0; tiltOutput = 0;
    tiltPID.SetMode(AUTOMATIC);

  } else {
    // --- 正常追蹤 ---
    digitalWrite(STEPPER_ENABLE_PIN, LOW); // 致能馬達

    // 只有在收到 "兩組" 新資料時才更新誤差
    // 這能確保 PID 的 Input 是同步的
    if (newMonkeyData && newSpotData) {
      panError = targetMonkeyX - laserSpotX;
      tiltError = targetMonkeyY - laserSpotY; // 根據Y軸方向，可能需改為 laserSpotY - targetMonkeyY
      
      newMonkeyData = false;
      newSpotData = false;
      
      Serial.printf("新誤差 => X: %.2f, Y: %.2f\n", panError, tiltError);
    }

    // 讓 PID 函式庫計算修正值
    panPID.Compute();
    tiltPID.Compute();

    // 將 PID 的輸出 (速度) 應用於馬達
    // panOutput 和 tiltOutput 會由 PID 函式庫自動更新
    panStepper.setSpeed(panOutput);
    tiltStepper.setSpeed(tiltOutput);
  }

  // AccelStepper 必須在 loop 中持續運行
  panStepper.runSpeed();
  tiltStepper.runSpeed();
}