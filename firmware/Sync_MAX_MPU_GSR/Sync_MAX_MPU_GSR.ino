// ESP32_GSR_MPU6050_MAX30102_Logger.ino
// Logs GSR (analog), MPU6050 accel, and MAX3010x IR with synchronized timestamps
// Outputs CSV to Serial and (optionally) Bluetooth SPP

#include <Wire.h>
#include "MAX30105.h"
#include "BluetoothSerial.h"

// ---------- PIN CONFIG ----------
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
const int GSR_PIN = 34;  // Must be ADC1 pin (32–39)

// ---------- MPU6050 CONFIG ----------
const uint8_t MPU_ADDR = 0x68;
const uint8_t ACCEL_XOUT_H = 0x3B;
const float ACC_SENS = 16384.0f; // ±2g

// ---------- Globals ----------
MAX30105 particleSensor;
BluetoothSerial SerialBT;
bool hasMAX = false;
char lineBuf[180];

// ---------- Helpers ----------
void initMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // Wake up
  Wire.endTransmission();
  delay(10);
}

bool readMPUAccel(float &ax_g, float &ay_g, float &az_g) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  const uint8_t toRead = 6;
  Wire.requestFrom((int)MPU_ADDR, (int)toRead, (uint8_t)true);
  if (Wire.available() < toRead) return false;
  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  ax_g = (float)raw_ax / ACC_SENS;
  ay_g = (float)raw_ay / ACC_SENS;
  az_g = (float)raw_az / ACC_SENS;
  return true;
}

bool initMAX() {
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    return false;
  }
  particleSensor.setup(31, 4, 2, 400, 411, 0x1F); // Tune as needed
  return true;
}

// ---------- GSR Reader ----------
int readGSRStable(int pin, int n=16) {
  (void)analogRead(pin);  // discard first sample (settling)
  delayMicroseconds(200);
  long sum = 0;
  for (int i = 0; i < n; i++) {
    sum += analogRead(pin);
    delay(3);
  }
  return sum / n;
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(1); }
  Serial.println("ESP32 GSR + MPU6050 + MAX3010x Logger starting...");

  // I2C init
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  delay(10);

  initMPU();
  hasMAX = initMAX();
  if (!hasMAX) Serial.println("Warning: MAX30105 not found or init failed.");

  SerialBT.begin("ESP32_Logger");
  Serial.println("Bluetooth SPP started (device name: ESP32_Logger).");

  analogSetPinAttenuation(GSR_PIN, ADC_11db);

  // CSV header
  Serial.println("ts_us,ax_g,ay_g,az_g,ir_raw,gsr_raw,gsr_v");
  if (SerialBT.connected()) {
    SerialBT.println("ts_us,ax_g,ay_g,az_g,ir_raw,gsr_raw,gsr_v");
  }
}

// ---------- LOOP ----------
void loop() {
  unsigned long ts_us = micros();

  // --- Read MPU6050 ---
  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  readMPUAccel(ax, ay, az);

  // --- Read MAX3010x ---
  uint32_t ir_raw = 0;
  if (hasMAX) {
    ir_raw = particleSensor.getIR();
  }

  // --- Read GSR ---
  int gsr_raw = readGSRStable(GSR_PIN, 10);
  float gsr_v = (gsr_raw / 4095.0f) * 3.3f;

  // --- Format CSV ---
  int len = snprintf(lineBuf, sizeof(lineBuf),
                     "%lu,%.5f,%.5f,%.5f,%lu,%d,%.3f\n",
                     ts_us, ax, ay, az,
                     (unsigned long)ir_raw,
                     gsr_raw, gsr_v);

  if (len > 0) {
    Serial.write((uint8_t*)lineBuf, (size_t)len);
    if (SerialBT.connected()) {
      SerialBT.write((uint8_t*)lineBuf, (size_t)len);
    }
  }

  delay(50); // ~20 Hz logging; adjust for your needs
}
