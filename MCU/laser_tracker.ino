/*
 * ===================================================================
 * ESP8266 雷射追蹤控制器
 * * 功能:
 * 1. 連接 WiFi 和 MQTT 伺服器。
 * 2. 訂閱 'tracker/target' (猴子座標) 和 'tracker/spot' (雷射光點座標)。
 * 3. 接收 JSON 格式的座標資料。
 * 4. 計算 猴子 和 光點 之間的像素誤差。
 * 5. 使用 P-Controller (比例控制) 將誤差轉換為步進馬達的修正步數。
 * 6. 透過 AccelStepper 函式庫驅動兩個步進馬達 (旋迴/Pan, 俯仰/Tilt)。
 *
 * 假設的 MQTT 訊息格式 (由您的 AI 伺服器發送):
 * * 主題: "tracker/target"
 * 內容: {"x": 320.5, "y": 240.0}  (mAP 最高的猴子座標)
 * * 主題: "tracker/spot"
 * 內容: {"x": 318.0, "y": 242.1}  (雷射光點座標)
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

// --- 1. WiFi 與 MQTT 設定 (請修改為您的資訊) ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "YOUR_MQTT_BROKER_IP"; // 例如: "192.168.1.100"
const int mqtt_port = 1883;
const char* mqtt_client_id = "ESP8266_LaserTracker";

// --- 2. 步進馬達腳位設定 (使用 DRV8825 或 A4988 驅動器) ---
// 旋迴 (Pan) 馬達
#define PAN_STEP_PIN D1 // GPIO 5
#define PAN_DIR_PIN  D2 // GPIO 4

// 俯仰 (Tilt) 馬達
#define TILT_STEP_PIN D5 // GPIO 14
#define TILT_DIR_PIN  D6 // GPIO 12

// 設定馬達介面為 1 (DRIVER 模式)
AccelStepper panStepper(AccelStepper::DRIVER, PAN_STEP_PIN, PAN_DIR_PIN);
AccelStepper tiltStepper(AccelStepper::DRIVER, TILT_STEP_PIN, TILT_DIR_PIN);

// --- 3. 全域變數 ---
WiFiClient espClient;
PubSubClient client(espClient);

// 座標狀態變數
float targetMonkeyX = 0.0;
float targetMonkeyY = 0.0;
float laserSpotX = 0.0;
float laserSpotY = 0.0;

// 旗標 (Flags)，用於標記是否收到新資料
volatile bool newMonkeyData = false;
volatile bool newSpotData = false;

// --- 4. 追蹤控制參數 (!! 關鍵調校參數 !!) ---
/*
 * Kp (Proportional Gain) 比例增益
 * 這是 "像素誤差" 到 "馬達步數" 的轉換因子。
 * - 如果 Kp 太小，馬達移動緩慢，追蹤不即時。
 * - 如果 Kp 太大，馬達會過度反應 (Overshoot)，在目標周圍抖動。
 * * 您必須透過實驗來調整這些值！
 */
float Kp_pan = 0.8;   // 旋迴 (X 軸) 的 P-gain
float Kp_tilt = -0.8; // 俯仰 (Y 軸) 的 P-gain
                      // 注意: Y 軸可能需要負號，這取決於您的相機和馬達安裝方向

// --- 5. WiFi 連線 ---
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

  Serial.println("");
  Serial.println("WiFi 已連線");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

// --- 6. MQTT 訊息回呼 (Callback) ---
/*
 * 這是核心功能。當 ESP8266 收到訂閱的 MQTT 訊息時，此函式會被觸發。
 */
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("收到訊息 [");
  Serial.print(topic);
  Serial.print("] ");

  // 將 payload 轉換為字串
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';
  Serial.println(message);

  // 使用 ArduinoJson 解析 JSON
  StaticJsonDocument<128> doc;
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.print("deserializeJson() 失敗: ");
    Serial.println(error.c_str());
    return;
  }

  // 根據主題更新對應的座標
  if (strcmp(topic, "tracker/target") == 0) {
    // 這是猴子座標
    // 假設 AI 伺服器已經處理了 "mAP 最大" 的邏輯
    targetMonkeyX = doc["x"];
    targetMonkeyY = doc["y"];
    newMonkeyData = true;
  } 
  else if (strcmp(topic, "tracker/spot") == 0) {
    // 這是雷射光點座標
    laserSpotX = doc["x"];
    laserSpotY = doc["y"];
    newSpotData = true;
  }
}

// --- 7. MQTT 重新連線 ---
void reconnect() {
  // 循環直到重新連上
  while (!client.connected()) {
    Serial.print("嘗試 MQTT 連線中...");
    // 嘗試連線
    if (client.connect(mqtt_client_id)) {
      Serial.println("已連線");
      // 訂閱主題
      client.subscribe("tracker/target");
      client.subscribe("tracker/spot");
      Serial.println("已訂閱 tracker/target 和 tracker/spot");
    } else {
      Serial.print("失敗, rc=");
      Serial.print(client.state());
      Serial.println(" 5 秒後重試");
      // 等待 5 秒
      delay(5000);
    }
  }
}

// --- 8. 計算並移動馬達 ---
/*
 * 這就是您提到的 "比對誤差" 並 "提供修正值" 的邏輯
 */
void calculateAndMove() {
  // 必須同時有猴子和光點的新資料才進行計算
  if (newMonkeyData && newSpotData) {
    
    // 1. 計算誤差 (Error)
    float errorX = targetMonkeyX - laserSpotX; // X 軸誤差 (像素)
    float errorY = targetMonkeyY - laserSpotY; // Y 軸誤差 (像素)

    // 2. 計算修正值 (P-Controller)
    // 修正值 = 增益 * 誤差
    long moveStepsX = (long)(Kp_pan * errorX);
    long moveStepsY = (long)(Kp_tilt * errorY);

    Serial.printf("誤差 X: %.2f, Y: %.2f | 修正步數 X: %ld, Y: %ld\n", 
                  errorX, errorY, moveStepsX, moveStepsY);

    // 3. 驅動步進馬達
    // .move() 會在目前位置 *加上* 這些步數
    // AccelStepper 會在背景中自動處理加減速
    panStepper.move(moveStepsX);
    tiltStepper.move(moveStepsY);

    // 4. 重設旗標，等待下一組新資料
    newMonkeyData = false;
    newSpotData = false;
  }
}

// --- 9. Arduino Setup ---
void setup() {
  Serial.begin(115200);

  // 設定馬達參數
  // 這些值也需要根據您的馬達和驅動器進行調校
  panStepper.setMaxSpeed(1000.0);      // 旋迴馬達最大速度 (步/秒)
  panStepper.setAcceleration(500.0);   // 旋迴馬達加速度 (步/秒^2)
  
  tiltStepper.setMaxSpeed(1000.0);     // 俯仰馬達最大速度
  tiltStepper.setAcceleration(500.0);  // 俯仰馬達加速度

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

// --- 10. Arduino Loop ---
void loop() {
  // 保持 WiFi 和 MQTT 連線
  if (WiFi.status() != WL_CONNECTED) {
    setup_wifi();
  }
  if (!client.connected()) {
    reconnect();
  }
  client.loop(); // 處理 MQTT 訊息 (這會觸發 callback)

  // 檢查是否需要計算並移動
  calculateAndMove();

  // 持續運行馬達
  // .run() 是非阻塞的。它會根據需要移動一步。
  // 必須在 loop 中盡可能頻繁地呼叫它們。
  panStepper.run();
  tiltStepper.run();
}