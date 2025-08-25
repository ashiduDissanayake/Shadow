import Foundation
import CoreBluetooth
import Combine

@MainActor
final class BLEManager: NSObject, ObservableObject {
    // Published UI state
    @Published var isBluetoothPoweredOn = false
    @Published var isScanning = false
    @Published var foundDevices: [CBPeripheral] = []
    @Published var connectedPeripheral: CBPeripheral?
    @Published var connectionStatus: String = "Disconnected"
    @Published var latestValue: Int?
    @Published var pairedDeviceIdentifier: UUID? =
        UserDefaults.standard.string(forKey: "pairedPeripheralUUID").flatMap(UUID.init(uuidString:))

    // BLE
    private var centralManager: CBCentralManager!
    private var notifyCharacteristic: CBCharacteristic?

    // Keep a stable instance per UUID
    private var peripheralsById: [UUID: CBPeripheral] = [:]

    // UUIDs
    let serviceUUID = CBUUID(string: "6e400001-b5a3-f393-e0a9-e50e24dcca9e")
    let characteristicUUID = CBUUID(string: "6e400003-b5a3-f393-e0a9-e50e24dcca9e")

    override init() {
        super.init()
        // Callbacks on main queue for safe UI updates
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }

    func startScanning() {
        guard isBluetoothPoweredOn else {
            connectionStatus = "Bluetooth not available"
            return
        }
        foundDevices.removeAll()
        isScanning = true
        connectionStatus = "Scanning…"

        // IMPORTANT: do NOT filter by service here, because the UUID may be in the scan response
        // and CoreBluetooth won’t match it. Scan all and filter manually.
        centralManager.scanForPeripherals(withServices: nil,
                                          options: [CBCentralManagerScanOptionAllowDuplicatesKey: false])
    }

    func stopScanning() {
        isScanning = false
        centralManager.stopScan()
        if connectionStatus.hasPrefix("Scanning") {
            connectionStatus = "Scan stopped"
        }
    }

    func connect(to peripheral: CBPeripheral) {
        guard isBluetoothPoweredOn else {
            connectionStatus = "Bluetooth not available"
            return
        }
        let target = peripheralsById[peripheral.identifier] ?? peripheral
        peripheralsById[target.identifier] = target
        connectedPeripheral = target
        notifyCharacteristic = nil
        connectionStatus = "Connecting to \(target.name ?? "ESP32")…"
        centralManager.connect(target, options: nil)
    }

    func disconnect() {
        if let p = connectedPeripheral {
            centralManager.cancelPeripheralConnection(p)
        }
        connectedPeripheral = nil
        notifyCharacteristic = nil
        connectionStatus = "Disconnected"
    }

    func forgetDevice() {
        let wasConnected = connectedPeripheral != nil
        disconnect()
        pairedDeviceIdentifier = nil
        UserDefaults.standard.removeObject(forKey: "pairedPeripheralUUID")
        connectionStatus = wasConnected ? "Device forgotten & disconnected" : "Device forgotten"
    }
}

extension BLEManager: CBCentralManagerDelegate, CBPeripheralDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            isBluetoothPoweredOn = true
            connectionStatus = "Bluetooth is ON"

            // If we have a paired UUID, try to retrieve and auto-connect
            if let id = pairedDeviceIdentifier {
                let retrieved = centralManager.retrievePeripherals(withIdentifiers: [id])
                if let p = retrieved.first {
                    peripheralsById[id] = p
                    connectionStatus = "Reconnecting to saved device…"
                    connect(to: p)
                } else {
                    // Not cached by the system yet; a normal scan will find it again
                    startScanning()
                }
            }
        case .unsupported:
            isBluetoothPoweredOn = false
            connectionStatus = "Bluetooth unsupported"
            stopScanning()
        case .unauthorized:
            isBluetoothPoweredOn = false
            connectionStatus = "Bluetooth unauthorized"
            stopScanning()
        case .poweredOff:
            isBluetoothPoweredOn = false
            connectionStatus = "Bluetooth is OFF"
            stopScanning()
        default:
            isBluetoothPoweredOn = false
            connectionStatus = "Bluetooth not available"
            stopScanning()
        }
    }

    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any],
                        rssi RSSI: NSNumber) {

        // Keep stable instance
        let id = peripheral.identifier
        let stored = peripheralsById[id] ?? peripheral
        peripheralsById[id] = stored

        // Filter: by name OR by advertised services if present
        let nameMatches = (stored.name ?? "").localizedCaseInsensitiveContains("ESP32")
        let advertisedServices = advertisementData[CBAdvertisementDataServiceUUIDsKey] as? [CBUUID] ?? []
        let serviceMatches = advertisedServices.contains(serviceUUID)

        if nameMatches || serviceMatches {
            if !foundDevices.contains(where: { $0.identifier == id }) {
                foundDevices.append(stored)
            }
        }

        // If paired, auto-connect when we see it
        if let pairedId = pairedDeviceIdentifier, pairedId == id {
            if connectedPeripheral?.identifier != pairedId {
                connect(to: stored)
            }
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        connectedPeripheral = peripheral
        connectionStatus = "Connected to \(peripheral.name ?? "ESP32")"
        peripheral.delegate = self
        stopScanning()
        peripheral.discoverServices([serviceUUID])

        // Save pairing on first success
        if pairedDeviceIdentifier == nil {
            pairedDeviceIdentifier = peripheral.identifier
            UserDefaults.standard.set(peripheral.identifier.uuidString, forKey: "pairedPeripheralUUID")
        }
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        connectionStatus = "Failed to connect\(error.map { ": \($0.localizedDescription)" } ?? "")"
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        connectedPeripheral = nil
        notifyCharacteristic = nil
        connectionStatus = error == nil ? "Disconnected" : "Disconnected: \(error!.localizedDescription)"
        // If we’re paired, we can auto-scan to find it again
        if pairedDeviceIdentifier != nil, isBluetoothPoweredOn {
            startScanning()
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard error == nil else {
            connectionStatus = "Service discovery error: \(error!.localizedDescription)"
            return
        }
        guard let services = peripheral.services else { return }
        for service in services where service.uuid == serviceUUID {
            peripheral.discoverCharacteristics([characteristicUUID], for: service)
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        guard error == nil else {
            connectionStatus = "Characteristic discovery error: \(error!.localizedDescription)"
            return
        }
        guard let characteristics = service.characteristics else { return }
        for c in characteristics where c.uuid == characteristicUUID {
            notifyCharacteristic = c
            peripheral.setNotifyValue(true, for: c)
            connectionStatus = "Subscribed to notifications"
        }
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        guard error == nil else {
            connectionStatus = "Update error: \(error!.localizedDescription)"
            return
        }
        guard characteristic.uuid == characteristicUUID,
              let data = characteristic.value,
              data.count >= 4 else { return }

        // Decode 32-bit little-endian from ESP32
        let int32Value = data.withUnsafeBytes { $0.load(as: Int32.self).littleEndian }
        latestValue = Int(int32Value)
        connectionStatus = "Latest value: \(latestValue ?? 0)"
    }
}
