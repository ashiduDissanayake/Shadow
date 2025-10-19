# Task 9: macOS Monitoring Application - Quick Reference Guide

**Device**: Shadow-9026  
**Service UUIDs**: 0xA000 (Stress), 0xB000 (Pairing)  
**Target**: macOS 13.0+ (SwiftUI + CoreBluetooth)  

---

## 🎯 **PAIRING PROTOCOL - CLIENT IMPLEMENTATION**

### **BLE Service UUIDs**

```swift
// Service UUIDs
let STRESS_SERVICE_UUID    = CBUUID(string: "0000A000-0000-1000-8000-00805F9B34FB")
let PAIRING_SERVICE_UUID   = CBUUID(string: "0000B000-0000-1000-8000-00805F9B34FB")

// Pairing Characteristics
let DEVICE_INFO_UUID       = CBUUID(string: "0000B001-0000-1000-8000-00805F9B34FB")  // READ
let PAIRING_STATE_UUID     = CBUUID(string: "0000B002-0000-1000-8000-00805F9B34FB")  // READ, NOTIFY
let PAIRING_CONTROL_UUID   = CBUUID(string: "0000B003-0000-1000-8000-00805F9B34FB")  // WRITE
let SECURITY_CHALLENGE_UUID = CBUUID(string: "0000B004-0000-1000-8000-00805F9B34FB") // READ, WRITE

// Stress Characteristic
let STRESS_STATE_UUID      = CBUUID(string: "0000A001-0000-1000-8000-00805F9B34FB")  // READ, NOTIFY
```

---

## 📝 **DATA STRUCTURES**

### **1. Device Info (0xB001) - READ**
```swift
struct DeviceInfo {
    let deviceID: Data           // 16 bytes - Shadow UUID
    let deviceName: String       // 32 bytes - "Shadow-9026"
    let firmwareVersion: String  // 16 bytes - "v1.0.0"
    let hardwareRevision: String // 16 bytes - "ESP32-S3"
    
    init(from data: Data) {
        deviceID = data.subdata(in: 0..<16)
        
        let nameData = data.subdata(in: 16..<48)
        deviceName = String(data: nameData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
        
        let fwData = data.subdata(in: 48..<64)
        firmwareVersion = String(data: fwData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
        
        let hwData = data.subdata(in: 64..<80)
        hardwareRevision = String(data: hwData, encoding: .utf8)?
            .trimmingCharacters(in: .controlCharacters.union(.whitespaces)) ?? "Unknown"
    }
}
```

### **2. Pairing State (0xB002) - READ/NOTIFY**
```swift
enum PairingState: UInt8 {
    case idle = 0
    case advertising = 1
    case connected = 2
    case pending = 3
    case paired = 4
    case rejected = 5
}

struct PairingStateInfo {
    let state: PairingState
    let pairedCount: UInt8
    let maxPaired: UInt8
    
    init(from data: Data) {
        state = PairingState(rawValue: data[0]) ?? .idle
        pairedCount = data[1]
        maxPaired = data[2]
    }
}
```

### **3. Pairing Commands (0xB003) - WRITE**
```swift
enum PairingCommand: UInt8 {
    case pairRequest = 1
    case unpair = 2
    case clearAll = 3
}

func writePairRequest(to characteristic: CBCharacteristic, 
                     peripheral: CBPeripheral) {
    let command = Data([PairingCommand.pairRequest.rawValue])
    peripheral.writeValue(command, for: characteristic, type: .withResponse)
}

func writeUnpair(deviceID: Data, 
                to characteristic: CBCharacteristic,
                peripheral: CBPeripheral) {
    var command = Data([PairingCommand.unpair.rawValue])
    command.append(deviceID)  // 16 bytes
    peripheral.writeValue(command, for: characteristic, type: .withResponse)
}
```

### **4. Security Challenge (0xB004) - READ/WRITE**

**READ (from Shadow)**:
```swift
struct SecurityChallenge {
    let challenge: Data      // 16 bytes - random challenge
    let timestamp: UInt64    // 8 bytes - microseconds
    
    init(from data: Data) {
        challenge = data.subdata(in: 0..<16)
        timestamp = data.subdata(in: 16..<24).withUnsafeBytes { $0.load(as: UInt64.self) }
    }
}
```

**WRITE (to Shadow)**:
```swift
func prepareChallengeResponse(challenge: Data, 
                              shadowDeviceID: Data,
                              clientDeviceID: Data,
                              clientName: String) -> Data {
    // 1. Compute SHA-256 response
    var hashInput = Data()
    hashInput.append(challenge)           // 16 bytes
    hashInput.append(shadowDeviceID)      // 16 bytes
    
    var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
    hashInput.withUnsafeBytes {
        _ = CC_SHA256($0.baseAddress, CC_LONG(hashInput.count), &hash)
    }
    
    let response = Data(hash.prefix(16))  // First 16 bytes
    
    // 2. Prepare write data
    var writeData = Data()
    writeData.append(response)            // 16 bytes - SHA-256 hash
    writeData.append(clientDeviceID)      // 16 bytes - client UUID
    writeData.append(clientName.data(using: .utf8) ?? Data())  // Variable length
    
    return writeData
}
```

### **5. Stress State (0xA001) - READ/NOTIFY**
```swift
struct StressState {
    let sequence: UInt8      // Bits 7-1: sequence number (0-127)
    let isStressed: Bool     // Bit 0: stress state (0=NORMAL, 1=STRESS)
    
    init(from data: Data) {
        let byte = data[0]
        sequence = byte >> 1          // Upper 7 bits
        isStressed = (byte & 0x01) == 1  // Lower 1 bit
    }
}
```

---

## 🔄 **PAIRING WORKFLOW - SWIFT IMPLEMENTATION**

### **Complete Pairing Function**

```swift
class PairingManager: ObservableObject {
    @Published var isPaired: Bool = false
    @Published var pairingState: PairingState = .idle
    
    private var peripheral: CBPeripheral?
    private var characteristics: [CBUUID: CBCharacteristic] = [:]
    
    func performPairing() async throws {
        guard let peripheral = peripheral else {
            throw PairingError.noPeripheral
        }
        
        // Step 1: Read device info
        guard let deviceInfoChar = characteristics[DEVICE_INFO_UUID] else {
            throw PairingError.characteristicNotFound
        }
        
        let deviceInfoData = try await readCharacteristic(deviceInfoChar, from: peripheral)
        let deviceInfo = DeviceInfo(from: deviceInfoData)
        
        print("📱 Shadow Device: \(deviceInfo.deviceName)")
        print("🆔 Device ID: \(deviceInfo.deviceID.hexString)")
        print("🔧 Firmware: \(deviceInfo.firmwareVersion)")
        print("⚙️ Hardware: \(deviceInfo.hardwareRevision)")
        
        // Step 2: Send pair request
        guard let pairingControlChar = characteristics[PAIRING_CONTROL_UUID] else {
            throw PairingError.characteristicNotFound
        }
        
        let pairCommand = Data([PairingCommand.pairRequest.rawValue])
        try await writeCharacteristic(pairingControlChar, 
                                      value: pairCommand, 
                                      to: peripheral)
        
        print("📤 Sent pairing request")
        
        // Step 3: Wait for pairing state to change to PENDING
        try await waitForPairingState(.pending, timeout: 5.0)
        
        // Step 4: Read challenge
        guard let securityChallengeChar = characteristics[SECURITY_CHALLENGE_UUID] else {
            throw PairingError.characteristicNotFound
        }
        
        let challengeData = try await readCharacteristic(securityChallengeChar, from: peripheral)
        let securityChallenge = SecurityChallenge(from: challengeData)
        
        print("🔐 Received challenge (timestamp: \(securityChallenge.timestamp))")
        
        // Step 5: Compute response
        let clientDeviceID = getOrCreateClientDeviceID()  // Generate/retrieve UUID
        let clientName = Host.current().localizedName ?? "Mac"
        
        let responseData = prepareChallengeResponse(
            challenge: securityChallenge.challenge,
            shadowDeviceID: deviceInfo.deviceID,
            clientDeviceID: clientDeviceID,
            clientName: clientName
        )
        
        // Step 6: Send response
        try await writeCharacteristic(securityChallengeChar, 
                                      value: responseData, 
                                      to: peripheral)
        
        print("📤 Sent challenge response")
        
        // Step 7: Wait for pairing state to change to PAIRED
        try await waitForPairingState(.paired, timeout: 5.0)
        
        print("✅ Pairing successful!")
        isPaired = true
        
        // Save pairing info to UserDefaults
        savePairingInfo(deviceInfo: deviceInfo, clientDeviceID: clientDeviceID)
    }
    
    private func waitForPairingState(_ targetState: PairingState, 
                                     timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        
        while Date() < deadline {
            if pairingState == targetState {
                return
            }
            
            if pairingState == .rejected {
                throw PairingError.pairingRejected
            }
            
            try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        }
        
        throw PairingError.timeout
    }
    
    private func getOrCreateClientDeviceID() -> Data {
        let key = "ClientDeviceID"
        
        if let existingID = UserDefaults.standard.data(forKey: key) {
            return existingID
        }
        
        // Generate new UUID (16 bytes)
        var uuid = UUID().uuid
        let data = Data(bytes: &uuid, count: MemoryLayout.size(ofValue: uuid))
        
        UserDefaults.standard.set(data, forKey: key)
        return data
    }
    
    private func savePairingInfo(deviceInfo: DeviceInfo, clientDeviceID: Data) {
        let pairing = [
            "shadowDeviceID": deviceInfo.deviceID,
            "shadowDeviceName": deviceInfo.deviceName,
            "clientDeviceID": clientDeviceID,
            "pairTimestamp": Date()
        ] as [String : Any]
        
        UserDefaults.standard.set(pairing, forKey: "PairingInfo_\(deviceInfo.deviceName)")
    }
}

enum PairingError: Error {
    case noPeripheral
    case characteristicNotFound
    case timeout
    case pairingRejected
}
```

---

## 📊 **STRESS DATA VISUALIZATION**

### **Subscribe to Stress Notifications**

```swift
class StressMonitor: ObservableObject {
    @Published var currentStressLevel: Double = 0.0
    @Published var isStressed: Bool = false
    @Published var sequenceNumber: UInt8 = 0
    
    func subscribeToStressUpdates(peripheral: CBPeripheral, 
                                 characteristic: CBCharacteristic) {
        peripheral.setNotifyValue(true, for: characteristic)
    }
    
    func handleStressUpdate(data: Data) {
        let stressState = StressState(from: data)
        
        self.sequenceNumber = stressState.sequence
        self.isStressed = stressState.isStressed
        
        // Estimate stress level (0-100%)
        // This is a simplified mapping - you may want to add more logic
        self.currentStressLevel = stressState.isStressed ? 70.0 : 30.0
        
        print("📈 Stress Update #\(stressState.sequence): \(isStressed ? "STRESS" : "NORMAL") (\(currentStressLevel)%)")
    }
}
```

### **SwiftUI Stress Gauge**

```swift
struct StressGaugeView: View {
    let stressLevel: Double  // 0-100
    
    var gaugeColor: Color {
        switch stressLevel {
        case 0..<30: return .green
        case 30..<50: return .yellow
        case 50..<70: return .orange
        default: return .red
        }
    }
    
    var body: some View {
        VStack {
            Text("Stress Level")
                .font(.headline)
            
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.2), lineWidth: 20)
                
                Circle()
                    .trim(from: 0, to: stressLevel / 100.0)
                    .stroke(gaugeColor, style: StrokeStyle(lineWidth: 20, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 1.0), value: stressLevel)
                
                Text("\(Int(stressLevel))%")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundColor(gaugeColor)
            }
            .frame(width: 200, height: 200)
            
            Text(stressLevel >= 50 ? "STRESS" : "NORMAL")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(gaugeColor)
        }
        .padding()
    }
}
```

---

## 🔍 **DEVICE DISCOVERY**

### **BLE Central Manager**

```swift
class BLEManager: NSObject, ObservableObject {
    private var centralManager: CBCentralManager!
    
    @Published var discoveredDevices: [UUID: CBPeripheral] = [:]
    @Published var connectedDevice: CBPeripheral?
    @Published var isScanning: Bool = false
    
    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }
    
    func startScanning() {
        guard centralManager.state == .poweredOn else {
            print("❌ Bluetooth not powered on")
            return
        }
        
        print("🔍 Scanning for Shadow devices...")
        
        // Scan for devices advertising pairing service
        centralManager.scanForPeripherals(
            withServices: [PAIRING_SERVICE_UUID],
            options: [CBCentralManagerScanOptionAllowDuplicatesKey: false]
        )
        
        isScanning = true
    }
    
    func stopScanning() {
        centralManager.stopScan()
        isScanning = false
        print("⏸️ Stopped scanning")
    }
    
    func connect(to peripheral: CBPeripheral) {
        print("🔗 Connecting to \(peripheral.name ?? "Unknown")...")
        peripheral.delegate = self
        centralManager.connect(peripheral, options: nil)
    }
    
    func disconnect() {
        guard let device = connectedDevice else { return }
        centralManager.cancelPeripheralConnection(device)
    }
}

// MARK: - CBCentralManagerDelegate
extension BLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("✅ Bluetooth powered on")
        case .poweredOff:
            print("❌ Bluetooth powered off")
        case .unauthorized:
            print("⚠️ Bluetooth unauthorized")
        default:
            print("ℹ️ Bluetooth state: \(central.state)")
        }
    }
    
    func centralManager(_ central: CBCentralManager, 
                       didDiscover peripheral: CBPeripheral,
                       advertisementData: [String : Any],
                       rssi RSSI: NSNumber) {
        
        let name = peripheral.name ?? "Unknown"
        print("📡 Discovered: \(name) (RSSI: \(RSSI))")
        
        discoveredDevices[peripheral.identifier] = peripheral
    }
    
    func centralManager(_ central: CBCentralManager, 
                       didConnect peripheral: CBPeripheral) {
        print("✅ Connected to \(peripheral.name ?? "Unknown")")
        
        connectedDevice = peripheral
        stopScanning()
        
        // Discover services
        peripheral.discoverServices([PAIRING_SERVICE_UUID, STRESS_SERVICE_UUID])
    }
    
    func centralManager(_ central: CBCentralManager,
                       didDisconnectPeripheral peripheral: CBPeripheral,
                       error: Error?) {
        print("🔌 Disconnected from \(peripheral.name ?? "Unknown")")
        connectedDevice = nil
        
        if let error = error {
            print("❌ Disconnect error: \(error.localizedDescription)")
        }
    }
}

// MARK: - CBPeripheralDelegate
extension BLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral, 
                   didDiscoverServices error: Error?) {
        guard error == nil else {
            print("❌ Service discovery error: \(error!.localizedDescription)")
            return
        }
        
        guard let services = peripheral.services else { return }
        
        for service in services {
            print("🔧 Discovered service: \(service.uuid)")
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                   didDiscoverCharacteristicsFor service: CBService,
                   error: Error?) {
        guard error == nil else {
            print("❌ Characteristic discovery error: \(error!.localizedDescription)")
            return
        }
        
        guard let characteristics = service.characteristics else { return }
        
        for characteristic in characteristics {
            print("📋 Discovered characteristic: \(characteristic.uuid)")
            
            // Subscribe to notifications if supported
            if characteristic.properties.contains(.notify) {
                peripheral.setNotifyValue(true, for: characteristic)
                print("🔔 Subscribed to notifications for \(characteristic.uuid)")
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                   didUpdateValueFor characteristic: CBCharacteristic,
                   error: Error?) {
        guard error == nil, let data = characteristic.value else {
            if let error = error {
                print("❌ Read error: \(error.localizedDescription)")
            }
            return
        }
        
        // Handle different characteristics
        switch characteristic.uuid {
        case PAIRING_STATE_UUID:
            let pairingState = PairingStateInfo(from: data)
            print("🔐 Pairing state updated: \(pairingState.state)")
            
        case STRESS_STATE_UUID:
            let stressState = StressState(from: data)
            print("📈 Stress update #\(stressState.sequence): \(stressState.isStressed ? "STRESS" : "NORMAL")")
            
        default:
            print("📦 Data received for \(characteristic.uuid): \(data.hexString)")
        }
    }
}
```

---

## 🛠️ **UTILITY EXTENSIONS**

### **Data to Hex String**
```swift
extension Data {
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}
```

### **SHA-256 Helper**
```swift
import CommonCrypto

func sha256(data: Data) -> Data {
    var hash = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
    data.withUnsafeBytes {
        _ = CC_SHA256($0.baseAddress, CC_LONG(data.count), &hash)
    }
    return Data(hash)
}
```

---

## 📱 **SAMPLE UI**

### **Main Content View**
```swift
struct ContentView: View {
    @StateObject private var bleManager = BLEManager()
    @StateObject private var pairingManager = PairingManager()
    @StateObject private var stressMonitor = StressMonitor()
    
    var body: some View {
        NavigationSplitView {
            // Sidebar: Device List
            List(selection: $bleManager.connectedDevice) {
                Section("Discovered Devices") {
                    ForEach(Array(bleManager.discoveredDevices.values), id: \.identifier) { device in
                        DeviceRowView(device: device, bleManager: bleManager)
                    }
                }
            }
            .navigationTitle("Shadow Devices")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(bleManager.isScanning ? "Stop" : "Scan") {
                        if bleManager.isScanning {
                            bleManager.stopScanning()
                        } else {
                            bleManager.startScanning()
                        }
                    }
                }
            }
            
        } detail: {
            // Detail: Monitoring Dashboard
            if let device = bleManager.connectedDevice {
                MonitoringDashboardView(
                    device: device,
                    pairingManager: pairingManager,
                    stressMonitor: stressMonitor
                )
            } else {
                Text("Select a device to monitor")
                    .font(.title2)
                    .foregroundColor(.secondary)
            }
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}

struct DeviceRowView: View {
    let device: CBPeripheral
    let bleManager: BLEManager
    
    var body: some View {
        HStack {
            Image(systemName: "sensor.fill")
                .foregroundColor(.blue)
            
            VStack(alignment: .leading) {
                Text(device.name ?? "Unknown")
                    .font(.headline)
                Text(device.identifier.uuidString)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Button("Connect") {
                bleManager.connect(to: device)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.vertical, 4)
    }
}

struct MonitoringDashboardView: View {
    let device: CBPeripheral
    @ObservedObject var pairingManager: PairingManager
    @ObservedObject var stressMonitor: StressMonitor
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text(device.name ?? "Unknown")
                    .font(.title)
                    .fontWeight(.bold)
                
                Spacer()
                
                if pairingManager.isPaired {
                    Label("Paired", systemImage: "checkmark.shield.fill")
                        .foregroundColor(.green)
                } else {
                    Button("Pair Device") {
                        Task {
                            try? await pairingManager.performPairing()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            
            Divider()
            
            // Stress Monitoring
            if pairingManager.isPaired {
                StressGaugeView(stressLevel: stressMonitor.currentStressLevel)
                
                // Sensor Graphs
                SensorGraphsView(stressMonitor: stressMonitor)
                
            } else {
                Text("Device must be paired to view stress data")
                    .font(.title3)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding()
    }
}
```

---

## ✅ **TESTING CHECKLIST**

### **BLE Connection**
- [ ] macOS Bluetooth enabled
- [ ] App has Bluetooth permissions
- [ ] Shadow device advertising (Shadow-9026 visible)
- [ ] Connection established
- [ ] Services discovered (0xA000, 0xB000)
- [ ] Characteristics discovered (all 5)

### **Pairing Flow**
- [ ] Read device info successfully
- [ ] Send pair request
- [ ] Receive challenge
- [ ] Compute correct SHA-256 response
- [ ] Write response + client info
- [ ] Pairing state changes to PAIRED
- [ ] Pairing persists after app restart

### **Stress Monitoring**
- [ ] Subscribe to stress notifications
- [ ] Receive updates every 60 seconds
- [ ] Parse sequence number correctly
- [ ] Parse stress state correctly (NORMAL/STRESS)
- [ ] UI updates in real-time

### **Multi-Device**
- [ ] Pair 3 different Macs
- [ ] Verify 3/3 limit
- [ ] Unpair device works
- [ ] Clear all works

---

## 🚀 **QUICK START COMMANDS**

### **Create Xcode Project**
```bash
# Create new macOS app
cd ~/Dev/Shadow
mkdir ShadowMonitor
cd ShadowMonitor

# Open Xcode
open -a Xcode

# Create new macOS App project:
# - Name: ShadowMonitor
# - Interface: SwiftUI
# - Language: Swift
# - Minimum macOS: 13.0
```

### **Add Capabilities**
1. Select project in Xcode
2. Select target → Signing & Capabilities
3. Add "Bluetooth" capability
4. Add "Keychain Sharing" (for storing device UUID)

### **Update Info.plist**
```xml
<key>NSBluetoothAlwaysUsageDescription</key>
<string>Shadow Monitor needs Bluetooth to connect to your stress monitoring device</string>
```

---

## 🎯 **EXPECTED DEVICE INFO**

When you read Device Info characteristic (0xB001), you should receive:

```
Device ID:        9251B891...EF3D9026 (16 bytes)
Device Name:      Shadow-9026 (32 bytes, null-terminated)
Firmware Version: v1.0.0 (16 bytes, null-terminated)
Hardware Revision: ESP32-S3 (16 bytes, null-terminated)

Total: 80 bytes
```

---

## 📌 **IMPORTANT NOTES**

1. **Challenge Timeout**: Challenge expires after 30 seconds. Complete pairing within this window.

2. **SHA-256 Computation**: Must hash `challenge + shadow_device_id` (NOT `challenge + client_device_id`).

3. **Multi-Device Limit**: Maximum 3 paired devices. Unpair old devices if limit reached.

4. **NVS Persistence**: Pairings survive ESP32 reboots. Test by power cycling Shadow device.

5. **Stress Updates**: Sent every 60 seconds. Subscribe to notifications for real-time updates.

6. **Sequence Numbers**: Use to detect missed packets (should increment 0→127 then wrap to 0).

---

**Ready to build the macOS app!** 🚀
