/*
 * ===================================================================
 * ESP8266 雷射致動器 (Actuator)
 *
 * 理論基礎 (依據 PDF 文件):
 * 1. 本程式為 "開迴路" 控制，不計算誤差。
 * 2. 所有 "修正值" (即絕對角度) 均由伺服器計算 (使用校正函式 F)。
 * 3. ESP8266 僅作為 MQTT 致動器，負責接收並執行絕對角度命令。
 *
 * 訂閱的 MQTT 主題:
 * 1. "laser/command/absolute_angle"
 * - 內容 (JSON): {"pan": 95.5, "tilt": 32.1}
 * - 來自伺服器的即時運作命令 。
 *
 * 2. "laser/command/fire_laser"
 * - 內容 (JSON): {"state": true} 或 {"state": false}
 * - 用於 "一鍵校正"  流程中，由伺服器控制雷射點亮。
 *
 * 3. "laser/command/rehome"
 * - 內容 (JSON): {}
 * - (可選) 用於重設馬達原點 (0, 0)。
 *
 * 發布的 MQTT 主題:
 * 1. "laser/status/motor"
 * - 內容 (String): "idle" 或 "moving"
 * - 回報馬達狀態，用於校正流程的同步。
 * ===================================================================
 
 必要函式庫：
 PubSubClient
 AccelStepper
 ArduinoJson
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>
#include <ArduinoJson.h>

// --- 1. WiFi 與 MQTT 設定 ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "YOUR_MQTT_BROKER_IP";
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP8266_LaserActuator";

// --- 2. 硬體腳位設定 ---
// 旋迴 (Pan) 馬達 (X 軸)
#define PAN_STEP_PIN D1 // GPIO 5
#define PAN_DIR_PIN  D2 // GPIO 4

// 俯仰 (Tilt) 馬達 (Y 軸)
#define TILT_STEP_PIN D5 // GPIO 14
#define TILT_DIR_PIN  D6 // GPIO 12

// 步進馬達驅動器致能 (Enable) 腳位 (低電位致能)
#define STEPPER_ENABLE_PIN D8 // GPIO 15

// 雷射模組控制腳位
#define LASER_PIN D7 // GPIO 13

// --- 3. 馬達物理參數 (!! 關鍵設定 !!) ---
/*
 * 您必須根據您的硬體計算此值
 * (馬達每轉步數 * 驅動器微步) / 360 度
 * 範例: (200 步 * 16 微步) / 360 = 8.888...
 */
const float STEPS_PER_DEGREE_PAN = 8.888;
const float STEPS_PER_DEGREE_TILT = 8.888;

// --- 4. 全域變數 ---
AccelStepper panStepper(AccelStepper::DRIVER, PAN_STEP_PIN, PAN_DIR_PIN);
AccelStepper tiltStepper(AccelStepper::DRIVER, TILT_STEP_PIN, TILT_DIR_PIN);

WiFiClient espClient;
PubSubClient client(espClient);

bool motor_is_moving = false;
unsigned long lastMoveTime = 0;
const long motorDisableTimeout = 5000; // 馬達停止 5 秒後禁用 (省電/降溫)

// --- 5. MQTT 訊息回呼 (Callback) ---
void callback(char* topic, byte* payload, unsigned int length) {
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

  // === 主題 1: 執行絕對角度移動 (即時運作) ===
  if (strcmp(topic, "laser/command/absolute_angle") == 0) {
    // 啟用馬達
    digitalWrite(STEPPER_ENABLE_PIN, LOW);
    motor_is_moving = true;
    client.publish("laser/status/motor", "moving");

    // 從 JSON 讀取伺服器計算好的 "修正值" (絕對角度)
    float pan_angle = doc["pan"];
    float tilt_angle = doc["tilt"];

    // 將 "角度" 轉換為 "絕對步數"
    long pan_steps = (long)(pan_angle * STEPS_PER_DEGREE_PAN);
    long tilt_steps = (long)(tilt_angle * STEPS_PER_DEGREE_TILT);

    Serial.printf("命令: Pan=%.2f°, Tilt=%.2f° | Steps=%ld, %ld\n", 
                  pan_angle, tilt_angle, pan_steps, tilt_steps);

    // 命令馬達移動到該絕對位置
    panStepper.moveTo(pan_steps);
    tiltStepper.moveTo(tilt_steps);
  }

  // === 主題 2: 控制雷射開關 (校正時使用) ===
  else if (strcmp(topic, "laser/command/fire_laser") == 0) {
    bool state = doc["state"];
    if (state) {
      digitalWrite(LASER_PIN, HIGH); // 打開雷射
      Serial.println("命令: 雷射開啟");
    } else {
      digitalWrite(LASER_PIN, LOW); // 關閉雷射
      Serial.println("命令: 雷射關閉");
    }
  }

  // === 主題 3: (可選) 馬達歸零 ===
  else if (strcmp(topic, "laser/command/rehome") == 0) {
    Serial.println("命令: 重設原點 (0, 0)");
    // 這裡可以添加尋找限位開關的歸零 (Homing) 程式碼
    // 目前僅作軟體歸零
    panStepper.setCurrentPosition(0);
    tiltStepper.setCurrentPosition(0);
    panStepper.moveTo(0);
    tiltStepper.moveTo(0);
  }
}

// --- 6. MQTT 重新連線 ---
void reconnect() {
  while (!client.connected()) {
    Serial.print("嘗試 MQTT 連線中...");
    if (client.connect(mqtt_client_id)) {
      Serial.println("已連線");
      // 訂閱所有命令主題
      client.subscribe("laser/command/absolute_angle");
      client.subscribe("laser/command/fire_laser");
      client.subscribe("laser/command/rehome");
    } else {
      Serial.printf("失敗, rc=%d, 5 秒後重試\n", client.state());
      delay(5000);
    }
  }
}

// --- 7. WiFi 連線 ---
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

// --- 8. Arduino Setup ---
void setup() {
  Serial.begin(115200);

  // 初始化腳位
  pinMode(LASER_PIN, OUTPUT);
  pinMode(STEPPER_ENABLE_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);      // 預設關閉雷射
  digitalWrite(STEPPER_ENABLE_PIN, HIGH); // 預設禁用馬達

  // 設定馬達參數 (速度和加速度)
  panStepper.setMaxSpeed(1000.0);
  panStepper.setAcceleration(500.0);
  
  tiltStepper.setMaxSpeed(1000.0);
  tiltStepper.setAcceleration(500.0);

  // 連線
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

// --- 9. Arduino Loop ---
void loop() {
  // 保持網路連線
  if (WiFi.status() != WL_CONNECTED) {
    setup_wifi();
  }
  if (!client.connected()) {
    reconnect();
  }
  client.loop(); // 處理 MQTT 訊息 (會觸發 callback)

  // 檢查馬達是否還在移動
  if (panStepper.distanceToGo() == 0 && tiltStepper.distanceToGo() == 0) {
    // 馬達剛剛停止
    if (motor_is_moving) {
      Serial.println("狀態: 馬達已停止 (Idle)");
      client.publish("laser/status/motor", "idle"); // 發布 "idle" 狀態
      motor_is_moving = false;
      lastMoveTime = millis(); // 開始計時
    }
  }

  // 如果馬達停止超過 5 秒，則禁用驅動器以節省電力並降溫
  if (!motor_is_moving && (millis() - lastMoveTime > motorDisableTimeout)) {
    digitalWrite(STEPPER_ENABLE_PIN, HIGH); // 禁用馬DA
  }

  // AccelStepper 必須在 loop 中持續運行以產生脈衝
  panStepper.run();
  tiltStepper.run();
}