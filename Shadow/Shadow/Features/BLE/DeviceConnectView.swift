import SwiftUI

struct DeviceConnectView: View {
    @StateObject var bleManager = BLEManager()
    
    var body: some View {
        VStack(spacing: 24) {
            Text("Connect to ESP32")
                .font(.title2.bold())
                .foregroundColor(.white)
            
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
                        .foregroundColor(.primary)
                    Spacer()
                    if bleManager.connectedPeripheral?.identifier == device.identifier {
                        Text("Connected").foregroundColor(.green)
                    } else {
                        Button("Connect") {
                            bleManager.connect(to: device)
                        }
                        .disabled(bleManager.connectedPeripheral != nil || !bleManager.isBluetoothPoweredOn)
                    }
                }
            }
            .frame(height: 200)
            
            Text("Status: \(bleManager.connectionStatus)")
                .foregroundColor(.white.opacity(0.7))
                .lineLimit(2)
                .minimumScaleFactor(0.6)
            
            if let value = bleManager.latestValue {
                Text("Latest ESP32 Value: \(value)")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.yellow)
            }
        }
        .padding()
        .background(
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.08, blue: 0.15), Color(red: 0.1, green: 0.15, blue: 0.25)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
        )
    }
}
