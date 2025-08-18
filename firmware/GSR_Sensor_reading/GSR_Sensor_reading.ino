const int GSR_PIN = 34; // ADC1 pin
void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(GSR_PIN, ADC_11db); // better full-scale range
}

int readStable(int pin, int n=16) {
  // Throw away first sample (settling for S/H cap)
  (void)analogRead(pin);
  delayMicroseconds(200);
  long sum = 0;
  for (int i = 0; i < n; i++) {
    sum += analogRead(pin);
    delay(3); // allow some settling + reduce noise coupling
  }
  return sum / n;
}

void loop() {
  int raw = readStable(GSR_PIN);
  float v_pin = (raw / 4095.0f) * 3.3f;
  Serial.print("raw="); Serial.print(raw);
  Serial.print("  Vpin="); Serial.println(v_pin, 3);
  delay(50);
}
