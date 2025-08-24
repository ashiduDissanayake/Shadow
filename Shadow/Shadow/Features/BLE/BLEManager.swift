import Foundation
import CoreBluetooth

class BLEManager: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published var isBluetoothPoweredOn = false
    @Published var isScanning = false
    @Published var foundDevices: [CBPeripheral] = []
    @Published var connectedPeripheral: CBPeripheral?
    @Published var connectionStatus: String = "Disconnected"
    @Published var latestValue: Int?
    
    private var centralManager: CBCentralManager!
    private var notifyCharacteristic: CBCharacteristic?
    
    // Use the same UUIDs as in your ESP32 code
    let serviceUUID = CBUUID(string: "6e400001-b5a3-f393-e0a9-e50e24dcca9e")
    let characteristicUUID = CBUUID(string: "6e400003-b5a3-f393-e0a9-e50e24dcca9e")
    
    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }
    
    // MARK: - CBCentralManagerDelegate
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn {
            isBluetoothPoweredOn = true
            connectionStatus = "Bluetooth is ON"
        } else {
            isBluetoothPoweredOn = false
            connectionStatus = "Bluetooth not available"
            stopScanning()
        }
    }
    
    func startScanning() {
        guard isBluetoothPoweredOn else {
            connectionStatus = "Bluetooth not available"
            return
        }
        foundDevices = []
        isScanning = true
        // Scan for all services to avoid missing devices due to UUID mismatch.
        centralManager.scanForPeripherals(withServices: nil, options: nil)
        connectionStatus = "Scanning..."
    }
    
    func stopScanning() {
        isScanning = false
        centralManager.stopScan()
        connectionStatus = "Scan stopped"
    }
    
    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                       advertisementData: [String : Any], rssi RSSI: NSNumber) {
        print("Discovered: \(peripheral.name ?? "Unknown") (\(peripheral.identifier))")
        if !foundDevices.contains(where: { $0.identifier == peripheral.identifier }) {
            foundDevices.append(peripheral)
        }
    }
    
    func connect(to peripheral: CBPeripheral) {
        guard isBluetoothPoweredOn else {
            connectionStatus = "Bluetooth not available"
            return
        }
        connectedPeripheral = nil
        notifyCharacteristic = nil
        centralManager.connect(peripheral, options: nil)
        connectionStatus = "Connecting..."
    }
    
    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectedPeripheral = peripheral
        connectionStatus = "Connected to \(peripheral.name ?? "ESP32")"
        peripheral.delegate = self
        peripheral.discoverServices([serviceUUID])
        stopScanning()
    }
    
    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        connectionStatus = "Failed to connect"
    }
    
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        connectionStatus = "Disconnected"
        connectedPeripheral = nil
        notifyCharacteristic = nil
    }
    
    // MARK: - CBPeripheralDelegate
    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let services = peripheral.services else { return }
        for service in services {
            if service.uuid == serviceUUID {
                peripheral.discoverCharacteristics([characteristicUUID], for: service)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let characteristics = service.characteristics else { return }
        for characteristic in characteristics {
            if characteristic.uuid == characteristicUUID {
                notifyCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic)
                connectionStatus = "Subscribed to notifications"
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if characteristic.uuid == characteristicUUID, let value = characteristic.value {
            let int32Value = value.withUnsafeBytes { $0.load(as: Int32.self) }
            latestValue = Int(int32Value)
            connectionStatus = "Latest value: \(latestValue ?? 0)"
        }
    }
}
