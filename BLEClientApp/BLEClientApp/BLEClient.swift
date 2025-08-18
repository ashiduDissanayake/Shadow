import Foundation
import CoreBluetooth

// Custom Service and Characteristic UUIDs
let ServiceUUID = CBUUID(string: "A000")
let DataCharacteristicUUID = CBUUID(string: "A001")
let ControlCharacteristicUUID = CBUUID(string: "A002")
let StatusCharacteristicUUID = CBUUID(string: "A003")

class BLEClient: NSObject, CBCentralManagerDelegate, CBPeripheralDelegate {

    var centralManager: CBCentralManager!
    var esp32Peripheral: CBPeripheral?
    var dataCharacteristic: CBCharacteristic?
    var controlCharacteristic: CBCharacteristic?
    var statusCharacteristic: CBCharacteristic?

    var onDataReceived: ((String) -> Void)?
    var onStatusReceived: ((String) -> Void)?
    var onConnected: (() -> Void)?
    var onDisconnected: (() -> Void)?

    override init() {
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: nil)
    }

    func startScanning() {
        print("Starting scan for peripherals...")
        centralManager.scanForPeripherals(withServices: [ServiceUUID], options: nil)
    }

    func stopScanning() {
        print("Stopping scan.")
        centralManager.stopScan()
    }

    func disconnect() {
        if let peripheral = esp32Peripheral {
            centralManager.cancelPeripheralConnection(peripheral)
        }
    }

    func writeControlCommand(command: String) {
        guard let peripheral = esp32Peripheral, let characteristic = controlCharacteristic else {
            print("Peripheral or Control Characteristic not available.")
            return
        }
        let data = command.data(using: .utf8)!
        peripheral.writeValue(data, for: characteristic, type: .withResponse)
        print("Sent control command: \(command)")
    }

    func writeData(dataString: String) {
        guard let peripheral = esp32Peripheral, let characteristic = dataCharacteristic else {
            print("Peripheral or Data Characteristic not available.")
            return
        }
        let data = dataString.data(using: .utf8)!
        peripheral.writeValue(data, for: characteristic, type: .withoutResponse)
        print("Sent data: \(dataString)")
    }

    // MARK: - CBCentralManagerDelegate

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            print("Central Manager is powered on.")
            startScanning()
        case .poweredOff:
            print("Central Manager is powered off.")
        case .resetting:
            print("Central Manager is resetting.")
        case .unauthorized:
            print("Central Manager is unauthorized.")
        case .unknown:
            print("Central Manager state is unknown.")
        case .unsupported:
            print("Central Manager is unsupported.")
        @unknown default:
            fatalError("A previously unknown central manager state occurred.")
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral, advertisementData: [String : Any], rssi RSSI: NSNumber) {
        print("Discovered Peripheral: \(peripheral.name ?? "Unknown")")
        if peripheral.name == "ESP32_BLE_Device" {
            print("Found ESP32_BLE_Device! Connecting...")
            centralManager.stopScan()
            esp32Peripheral = peripheral
            esp32Peripheral?.delegate = self
            centralManager.connect(peripheral, options: nil)
        }
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        print("Connected to \(peripheral.name ?? "Unknown")")
        onConnected?()
        peripheral.discoverServices([ServiceUUID])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        print("Failed to connect to \(peripheral.name ?? "Unknown"): \(error?.localizedDescription ?? "Unknown error")")
        esp32Peripheral = nil
        startScanning() // Try scanning again
    }

    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        print("Disconnected from \(peripheral.name ?? "Unknown"): \(error?.localizedDescription ?? "Unknown error")")
        onDisconnected?()
        esp32Peripheral = nil
        dataCharacteristic = nil
        controlCharacteristic = nil
        statusCharacteristic = nil
        startScanning() // Start scanning again to allow reconnection
    }

    // MARK: - CBPeripheralDelegate

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        if let error = error {
            print("Error discovering services: \(error.localizedDescription)")
            return
        }
        guard let services = peripheral.services else { return }
        for service in services {
            print("Discovered service: \(service.uuid)")
            if service.uuid == ServiceUUID {
                peripheral.discoverCharacteristics([DataCharacteristicUUID, ControlCharacteristicUUID, StatusCharacteristicUUID], for: service)
            }
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        if let error = error {
            print("Error discovering characteristics: \(error.localizedDescription)")
            return
        }
        guard let characteristics = service.characteristics else { return }
        for characteristic in characteristics {
            print("Discovered characteristic: \(characteristic.uuid)")
            if characteristic.uuid == DataCharacteristicUUID {
                dataCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic) // Subscribe to notifications
                print("Subscribed to Data Characteristic notifications.")
            } else if characteristic.uuid == ControlCharacteristicUUID {
                controlCharacteristic = characteristic
            } else if characteristic.uuid == StatusCharacteristicUUID {
                statusCharacteristic = characteristic
                peripheral.setNotifyValue(true, for: characteristic) // Subscribe to notifications
                print("Subscribed to Status Characteristic notifications.")
            }
        }

        // Request MTU update after discovering characteristics
        // Note: iOS/macOS typically handles MTU negotiation automatically upon connection,
        // but you can explicitly request it if needed. The system will choose the largest supported MTU.
        // peripheral.maximumWriteValueLength(for: .withoutResponse) gives the current MTU payload size.
        print("Current maximum write value length (without response): \(peripheral.maximumWriteValueLength(for: .withoutResponse))")
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("Error updating value for characteristic: \(error.localizedDescription)")
            return
        }
        guard let value = characteristic.value else { return }
        let stringValue = String(data: value, encoding: .utf8) ?? "Invalid Data"

        if characteristic.uuid == DataCharacteristicUUID {
            print("Received Data: \(stringValue)")
            onDataReceived?(stringValue)
        } else if characteristic.uuid == StatusCharacteristicUUID {
            print("Received Status: \(stringValue)")
            onStatusReceived?(stringValue)
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("Error writing value for characteristic: \(error.localizedDescription)")
            return
        }
        print("Successfully wrote value to characteristic: \(characteristic.uuid)")
    }

    func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        if let error = error {
            print("Error changing notification state: \(error.localizedDescription)")
            return
        }
        if characteristic.isNotifying {
            print("Notification began for: \(characteristic.uuid)")
        } else {
            print("Notification stopped for: \(characteristic.uuid)")
        }
    }
}


