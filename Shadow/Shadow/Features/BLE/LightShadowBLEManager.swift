import Foundation
import CoreBluetooth
import Combine

/// Minimal BLE manager focused only on: scan → detect sequence delta → connect → handshake → optional replay.
/// All persistence is delegated to StressDataRepository (no CoreData manager abstraction).
final class LightShadowBLEManager: NSObject, ObservableObject {
    
    // Public state for UI
    @Published var isScanning = false
    @Published var debugLog: [String] = []
    @Published var currentStatus: Status = .idle
    @Published var lastSyncDate: Date?
    @Published var activePeripheralID: UUID?
    @Published var advertisedSequence: UInt8 = 0
    @Published var lastKnownSequence: UInt8 = 0
    @Published var eventsReceivedThisSync: Int = 0
    
    enum Status: String {
        case idle, scanning, connecting, handshaking, replaying, upToDate, disconnecting, error
    }
    
    // BLE IDs
    private let serviceUUID = CBUUID(string: "1800")
    private let fsmCharUUID = CBUUID(string: "1801")
    private let eventCharUUID = CBUUID(string: "1802")
    private let ackCharUUID = CBUUID(string: "1803")
    private let controlCharUUID = CBUUID(string: "1804")
    
    // Internal
    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var fsmChar: CBCharacteristic?
    private var eventChar: CBCharacteristic?
    private var controlChar: CBCharacteristic?
    private var ackChar: CBCharacteristic?
    
    private var pendingAdvertisedSequence: UInt8 = 0
    private var expectedReplayTarget: UInt8 = 0
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
    }
    
    // MARK: - Public API
    
    func start() {
        guard central.state == .poweredOn else {
            log("Bluetooth not powered on yet.")
            return
        }
        log("Starting scan")
        currentStatus = .scanning
        isScanning = true
        central.scanForPeripherals(withServices: [serviceUUID],
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
    }
    
    func stop() {
        isScanning = false
        central.stopScan()
        currentStatus = .idle
        log("Scan stopped")
    }
    
    // MARK: - Core Logic
    
    private func handleAdvertisement(_ peripheral: CBPeripheral,
                                     data: Data,
                                     rssi: Int) {
        guard data.count == 1 else { return }
        let byte = data[0]
        let seq = (byte >> 1) & 0x7F
        let stateBit = byte & 0x01
        let lastStored = StressDataRepository.shared.lastKnownSequence(peripheralID: peripheral.identifier)
        
        advertisedSequence = seq
        lastKnownSequence = lastStored
        
        if seq != lastStored {
            log("Sequence delta detected (adv=\(seq) / stored=\(lastStored)) – connecting")
            pendingAdvertisedSequence = seq
            connect(peripheral)
        } else {
            // Optional: log occasionally
        }
        _ = stateBit // (You can surface current state later)
    }
    
    private func connect(_ p: CBPeripheral) {
        stop()
        currentStatus = .connecting
        eventsReceivedThisSync = 0
        peripheral = p
        activePeripheralID = p.identifier
        p.delegate = self
        central.connect(p, options: nil)
    }
    
    private func sendHandshake() {
        guard let controlChar, let peripheral else { return }
        currentStatus = .handshaking
        let packet = Data([0x04, pendingAdvertisedSequence]) // ACKNOWLEDGE_TRANSITION
        peripheral.writeValue(packet, for: controlChar, type: .withResponse)
        log("Sent handshake [04, \(pendingAdvertisedSequence)]")
    }
    
    private func handleFSMUpdate(_ data: Data) {
        guard data.count >= 2 else {
            log("FSM notify too small"); disconnect()
            return
        }
        let _fsmState = data[0]
        let eventLogSequence = data[1]
        let lastPersisted = StressDataRepository.shared.lastEventSequence()
        
        if eventLogSequence > lastPersisted {
            currentStatus = .replaying
            expectedReplayTarget = eventLogSequence
            requestReplay(from: lastPersisted + 1)
        } else {
            log("Up to date (device=\(eventLogSequence), local=\(lastPersisted))")
            finalizeSyncAndDisconnect()
        }
    }
    
    private func requestReplay(from start: UInt8) {
        guard let controlChar, let peripheral else { return }
        let packet = Data([0x01, start]) // REPLAY_FROM_SEQUENCE
        peripheral.writeValue(packet, for: controlChar, type: .withResponse)
        log("Requested replay from seq \(start)")
    }
    
    private func handleEventData(_ data: Data) {
        // TODO: Parse real stress_event_t. Placeholder: we treat first byte as sequence.
        guard data.count >= 1 else { return }
        let seq = data[0]
        eventsReceivedThisSync += 1
        // Persist minimal event
        StressDataRepository.shared.addStressEvent(peripheralID: peripheral!.identifier,
                                                   sequence: seq,
                                                   stressState: 0,
                                                   eventTimestamp: Date())
        log("Event indication seq=\(seq) stored")
        
        if seq == expectedReplayTarget {
            finalizeSyncAndDisconnect()
        }
    }
    
    private func finalizeSyncAndDisconnect() {
        guard let peripheral else { return }
        // Save device sequence
        StressDataRepository.shared.updateDeviceSequence(peripheralID: peripheral.identifier,
                                                         sequence: pendingAdvertisedSequence,
                                                         state: 0)
        lastSyncDate = Date()
        currentStatus = .upToDate
        log("Finalizing – stored sequence now \(pendingAdvertisedSequence)")
        disconnect()
    }
    
    private func disconnect() {
        if let p = peripheral {
            currentStatus = .disconnecting
            central.cancelPeripheralConnection(p)
        }
    }
    
    private func log(_ msg: String) {
        debugLog.append("[\(timestamp())] \(msg)")
        if debugLog.count > 120 { debugLog.removeFirst(debugLog.count - 120) }
    }
    
    private func timestamp() -> String {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss.SSS"
        return df.string(from: Date())
    }
}

// MARK: - CBCentralManagerDelegate

extension LightShadowBLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            log("Bluetooth ON")
            start()
        case .poweredOff:
            log("Bluetooth OFF")
            stop()
        default:
            break
        }
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any],
                        rssi RSSI: NSNumber) {
        if let serviceData = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data],
           let raw = serviceData[serviceUUID] {
            handleAdvertisement(peripheral, data: raw, rssi: RSSI.intValue)
        }
    }
    
    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        log("Connected \(peripheral.name ?? "Unknown")")
        peripheral.discoverServices([serviceUUID])
    }
    
    func centralManager(_ central: CBCentralManager,
                        didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        log("Failed connect: \(error?.localizedDescription ?? "Unknown")")
        currentStatus = .error
        start()
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        log("Disconnected")
        self.peripheral = nil
        fsmChar = nil; eventChar = nil; controlChar = nil; ackChar = nil
        if currentStatus != .upToDate {
            currentStatus = .idle
        }
        // Resume scan after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { self.start() }
    }
}

// MARK: - CBPeripheralDelegate

extension LightShadowBLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverServices error: Error?) {
        if let error { log("Service discovery error: \(error)"); disconnect(); return }
        guard let services = peripheral.services else { disconnect(); return }
        if let svc = services.first(where: { $0.uuid == serviceUUID }) {
            peripheral.discoverCharacteristics([fsmCharUUID,
                                                eventCharUUID,
                                                controlCharUUID,
                                                ackCharUUID], for: svc)
        } else {
            log("Stress service missing"); disconnect()
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let error { log("Char discovery error: \(error)"); disconnect(); return }
        service.characteristics?.forEach { c in
            switch c.uuid {
            case fsmCharUUID: fsmChar = c
            case eventCharUUID: eventChar = c
            case controlCharUUID: controlChar = c
            case ackCharUUID: ackChar = c
            default: break
            }
        }
        if let fsmChar { peripheral.setNotifyValue(true, for: fsmChar) }
        if let eventChar { peripheral.setNotifyValue(true, for: eventChar) }
        sendHandshake()
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error { log("Update error: \(error)"); return }
        guard let data = characteristic.value else { return }
        if characteristic.uuid == fsmCharUUID {
            handleFSMUpdate(data)
        } else if characteristic.uuid == eventCharUUID {
            handleEventData(data)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didWriteValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error { log("Write error: \(error)") }
    }
}
