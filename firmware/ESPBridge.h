#pragma once
#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>

class EspBridge {
public:
  using LineHandler = std::function<void(const String&)>;

  bool begin(const char* ssid, const char* pass,
             const char* bonjourService = "_espbridge._tcp",
             uint16_t fallbackPort = 5001,
             const char* fallbackHost = nullptr) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pass);
    Serial.print("WiFi connecting");
    for (int i=0; i<60 && WiFi.status()!=WL_CONNECTED; ++i) {
      delay(500); Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi failed");
      return false;
    }
    Serial.print("WiFi OK, IP: "); Serial.println(WiFi.localIP());

    // Try Bonjour discovery of Mac server
    IPAddress hostIP;
    uint16_t port = 0;

    if (MDNS.begin("esp32")) { // start mDNS responder (hostname 'esp32.local')
      int n = MDNS.queryService(bonjourService, "tcp");
      if (n > 0) {
        hostIP = MDNS.IP(0);
        port = MDNS.port(0);
      }
    }

    if (port == 0) {
      // fallback if discovery failed
      if (fallbackHost) {
        WiFi.hostByName(fallbackHost, hostIP);
      } else {
        hostIP = IPAddress(192,168,1,100); // change if needed
      }
      port = fallbackPort;
    }

    Serial.printf("Connecting to server %s:%u\n", hostIP.toString().c_str(), port);
    _client.setTimeout(2000);
    if (!_client.connect(hostIP, port)) {
      Serial.println("Server connect failed");
      return false;
    }

    // TCP keepalive (helps power + NATs)
    _client.setNoDelay(true);

    // Send hello
    sendJSON("{\"type\":\"hello\",\"name\":\"esp32\"}");
    return true;
  }

  void onLine(LineHandler cb) { _onLine = cb; }

  void loop() {
    while (_client.connected() && _client.available()) {
      String line = _client.readStringUntil('\n');
      if (_onLine) _onLine(line);
    }
  }

  void sendJSON(const String& json) {
    if (_client.connected()) {
      _client.print(json);
      _client.print('\n');
    }
  }

  // Convenience helpers
  void sendTelemetry(const String& name, float tempC) {
    String payload = String("{\"type\":\"telemetry\",\"name\":\"") + name +
                     String("\",\"temp\":") + String(tempC,1) + "}";
    sendJSON(payload);
  }

private:
  WiFiClient _client;
  LineHandler _onLine = nullptr;
};
