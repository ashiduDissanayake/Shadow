#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#define SERVICE_UUID        "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
#define CHARACTERISTIC_UUID "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
#define DEVICE_NAME         "ESP32-BLE"   // Short name helps fit UUID in ADV

BLECharacteristic* pCharacteristic = nullptr;
uint32_t value = 0; // 32-bit value for clean transfer

void setup() {
  Serial.begin(115200);

  BLEDevice::init(DEVICE_NAME);

  BLEServer* pServer = BLEDevice::createServer();
  BLEService* pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());

  // Initialize value and make sure first read has something sensible
  pCharacteristic->setValue((uint8_t*)&value, sizeof(value));

  pService->start();

  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);

  // Ensure the 128-bit service UUID is in the PRIMARY advertisement payload:
  pAdvertising->setScanResponse(false);      // Keep primary ADV for UUID
  // Optional: remove connection interval preference field to save space
  // on older ESP32 BLE lib versions:
  // pAdvertising->setMinPreferred(0x00);

  BLEDevice::startAdvertising();
  Serial.println("BLE GATT Server is advertising with notifications!");
}

void loop() {
  static unsigned long last = 0;
  if (millis() - last > 5000) {
    last = millis();
    value++; // increment the counter
    pCharacteristic->setValue((uint8_t*)&value, sizeof(value));
    pCharacteristic->notify();
    Serial.print("Notified value: "); Serial.println(value);
  }
}
