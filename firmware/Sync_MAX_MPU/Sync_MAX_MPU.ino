// ESP32_MPU6050_MAX30102_SerialLogger.ino
// Reads MPU6050 accel and MAX3010x IR (shared I2C), timestamps with micros(),
// prints CSV lines to Serial Monitor and (optionally) Bluetooth SPP.

#include <Wire.h>
#include "MAX30105.h"
#include "BluetoothSerial.h"

#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

const uint8_t MPU_ADDR = 0x68;
const uint8_t ACCEL_XOUT_H = 0x3B;
const float ACC_SENS = 16384.0f; // ±2g

MAX30105 particleSensor;
BluetoothSerial SerialBT;

bool hasMAX = false;

char lineBuf[160];

void initMPU() {
  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // wake
  Wire.endTransmission();
  delay(10);
}

bool readMPUAccel(float &ax_g, float &ay_g, float &az_g) {
  // Request accelerometer registers (6 bytes) in one transfer
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false; // repeated start failed
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
  // Settings tuned for reasonable sampling; adjust to your module
  particleSensor.setup(31, 4, 2, 400, 411, 0x1F);
  return true;
}

void setup() {
  // Use a high baud to reduce Serial blocking (increase if needed)
  Serial.begin(115200);
  while (!Serial) { delay(1); } // wait for Serial (optional)
  Serial.println("ESP32 MPU6050 + MAX3010x Serial Logger starting...");

  // Initialize I2C on chosen pins
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  delay(10);

  // Init sensors
  initMPU();
  hasMAX = initMAX();
  if (!hasMAX) Serial.println("Warning: MAX30105 not found or init failed.");

  // Start Bluetooth SPP (optional). If not needed, you can comment out.
  SerialBT.begin("ESP32_Logger");
  Serial.println("Bluetooth SPP started (device name: ESP32_Logger).");

  // Print CSV header
  Serial.println("ts_us,ax_g,ay_g,az_g,ir_raw");
  if (SerialBT.connected()) {
    SerialBT.println("ts_us,ax_g,ay_g,az_g,ir_raw");
  }
}

void loop() {
  // Timestamp as close as possible BEFORE the reads
  unsigned long ts_us = micros();

  // Read accelerometer (single-burst)
  float ax = 0.0f, ay = 0.0f, az = 0.0f;
  if (!readMPUAccel(ax, ay, az)) {
    // Leave zeros or handle error; keep loop fast
  }

  // Read one IR sample from MAX3010x if present
  uint32_t ir_raw = 0;
  if (hasMAX) {
    // getIR is a fast FIFO read provided by the library
    ir_raw = particleSensor.getIR();
  }

  // Format CSV line into fixed buffer (no dynamic allocation)
  int len = snprintf(lineBuf, sizeof(lineBuf), "%lu,%.5f,%.5f,%.5f,%lu\n",
                     ts_us, ax, ay, az, (unsigned long)ir_raw);
  if (len > 0) {
    // Write to Serial (blocking on baudrate) and optionally to Bluetooth if connected
    Serial.write((uint8_t*)lineBuf, (size_t)len);
    if (SerialBT.connected()) {
      SerialBT.write((uint8_t*)lineBuf, (size_t)len);
    }
  }

  // Small cooperative yield — keep as short as possible to minimize timestamp gaps.
  // You can remove or reduce this as needed, but leaving a tiny yield helps RTOS tasks.
  delayMicroseconds(100000); // 100 us pause; adjust or remove for faster sampling
}
