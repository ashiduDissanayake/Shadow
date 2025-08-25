import SwiftUI

struct DeviceConnectView: View {
    @StateObject var bleManager = BLEManager()

    var body: some View {
        VStack(spacing: 24) {
            Text("Connect to ESP32")
                .font(.title2.bold())
                .foregroundColor(.white)

            // Pairing UI
            if bleManager.pairedDeviceIdentifier == nil {
                Button(bleManager.isScanning ? "Stop Scanning" : "Start Scanning") {
                    bleManager.isScanning ? bleManager.stopScanning() : bleManager.startScanning()
                }
                .padding()
                .background(bleManager.isBluetoothPoweredOn ? Color.purple : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(12)
                .disabled(!bleManager.isBluetoothPoweredOn)

                List(bleManager.foundDevices, id: \.identifier) { device in
                    HStack {
                        Text(device.name ?? "Unknown Device")
                        Spacer()
                        Button("Pair") {
                            bleManager.connect(to: device)
                        }
                        .disabled(!bleManager.isBluetoothPoweredOn)
                    }
                }
                .frame(height: 220)

                Text("Select your ESP32 device to pair.")
                    .foregroundColor(.yellow)
            } else {
                // Already paired
                if bleManager.connectedPeripheral != nil {
                    Text("Connected to your ESP32!")
                        .foregroundColor(.green)
                    Button("Disconnect") {
                        bleManager.disconnect()
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 6)
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                } else {
                    Text("Waiting for ESP32 to advertise…")
                        .foregroundColor(.orange)
                    Button("Scan for my device") {
                        bleManager.startScanning()
                    }
                    .padding()
                    .background(bleManager.isBluetoothPoweredOn ? Color.purple : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    .disabled(!bleManager.isBluetoothPoweredOn)
                }

                Button("Forget Device") {
                    bleManager.forgetDevice()
                }
                .padding(.horizontal)
                .padding(.vertical, 6)
                .background(Color.red)
                .foregroundColor(.white)
                .cornerRadius(8)
                .padding(.top, 8)
            }

            Text("Status: \(bleManager.connectionStatus)")
                .foregroundColor(.white.opacity(0.8))
                .lineLimit(2)
                .minimumScaleFactor(0.6)

            if let value = bleManager.latestValue {
                Text("Latest ESP32 Value: \(value)")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.yellow)
            }

            Group {
                Text("Paired: \(bleManager.pairedDeviceIdentifier?.uuidString ?? "nil")")
                Text("Connected: \(bleManager.connectedPeripheral?.identifier.uuidString ?? "nil")")
            }
            .foregroundColor(.gray)
            .font(.caption)
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.08, blue: 0.15),
                         Color(red: 0.1, green: 0.15, blue: 0.25)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
        .onAppear {
            // Helpful UX: if not paired but Bluetooth is ON, start a scan
            if bleManager.pairedDeviceIdentifier == nil && bleManager.isBluetoothPoweredOn && !bleManager.isScanning {
                bleManager.startScanning()
            }
        }
    }
}
