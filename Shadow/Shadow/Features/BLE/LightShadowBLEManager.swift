import Foundation
import CoreBluetooth
import Combine

/// LightShadowBLEManager
/// Simplified protocol:
///  - Passive scan parses combined advertisement byte (sequence/state)
///  - If delta == 1 (and alwaysConnectOnChange == false) -> update locally (no connection)
///  - If delta > 1 OR forced -> connect, write lastKnownSequence, then read response
/// Missed events response format:
///   Byte0: currentSequence
///   Byte1: currentStableState (0=CALM,1=STRESS)
///   Byte2: missedCount (N)
///   Then N * 2 bytes: [seq_i, state_i] pairs (ascending, each state_i 0/1)
final class LightShadowBLEManager: NSObject, ObservableObject {
    
    // MARK: - UI Published
    @Published var status: Status = .idle
    @Published var lastKnownSequence: UInt8 = 0
    @Published var currentStableState: UInt8 = 0 // 0=CALM, 1=STRESS
    @Published var missedEvents: [(seq: UInt8, state: UInt8)] = []
    @Published var logLines: [String] = []
    @Published var isScanning = false
    
    enum Status: String {
        case idle, scanning, connecting, requestingMissed, upToDate, error
    }
    
    // MARK: - Configuration
    private let serviceUUID = CBUUID(string: "A000")
    private let eventCharUUID = CBUUID(string: "A002")
    
    /// Set to true if you want to force a connection even when delta == 1
    private let alwaysConnectOnChange = false
    
    // MARK: - CoreBluetooth
    private var central: CBCentralManager!
    private var activePeripheral: CBPeripheral?
    private var eventChar: CBCharacteristic?
    
    // MARK: - Internal State
    private var advertisedSequence: UInt8 = 0
    private var advertisedState: UInt8 = 0
    private var connectReasonDelta: UInt8 = 0
    private var lastConnectAttempt = Date.distantPast
    private let connectThrottle: TimeInterval = 1.5
    
    // MARK: - Persistence Keys
    private let sequenceKey = "Shadow_LastKnownSequence_V1"
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
        loadLastKnownSequence()
        log("Initialized. lastKnownSequence=\(lastKnownSequence)")
    }
    
    // MARK: - Public Control
    func start() {
        guard central.state == .poweredOn else {
            log("Cannot start: Bluetooth not powered on.")
            return
        }
        if isScanning { return }
        isScanning = true
        status = .scanning
        central.scanForPeripherals(withServices: nil,
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
        log("Scanning started (allow duplicates)")
    }
    
    func stop() {
        guard isScanning else { return }
        central.stopScan()
        isScanning = false
        status = .idle
        log("Scanning stopped")
    }
    
    // MARK: - Advertisement Handling
    private func handleAdvertisement(peripheral: CBPeripheral, serviceData: Data) {
        // Expect exactly 1 byte (combined)
        guard serviceData.count == 1 else { return }
        let combined = serviceData[0]
        let seq = (combined >> 1) & 0x7F
        let state = combined & 0x01
        
        // If unchanged → ignore
        if seq == lastKnownSequence { return }
        
        let delta = modularDelta(from: lastKnownSequence, to: seq)
        advertisedSequence = seq
        advertisedState = state
        connectReasonDelta = delta
        
        log("Advertisement change: advSeq=\(seq) advState=\(state) delta=\(delta)")
        
        if delta == 1 && !alwaysConnectOnChange {
            // Local fast update
            applySimpleUpdate(sequence: seq, state: state)
        } else {
            // Prevent rapid re-connect storms
            guard Date().timeIntervalSince(lastConnectAttempt) > connectThrottle else {
                log("Throttle: skipping connection attempt (too soon)")
                return
            }
            lastConnectAttempt = Date()
            connectTo(peripheral)
        }
    }
    
    private func modularDelta(from old: UInt8, to new: UInt8) -> UInt8 {
        if new >= old { return new - old }
        // Wrap-around (0..127)
        return (128 - old) + new
    }
    
    private func applySimpleUpdate(sequence: UInt8, state: UInt8) {
        lastKnownSequence = sequence
        currentStableState = state
        missedEvents.removeAll()
        saveSequence(sequence)
        status = .upToDate
        log("Applied simple update locally -> seq=\(sequence) state=\(state)")
    }
    
    // MARK: - Connection Flow
    private func connectTo(_ peripheral: CBPeripheral) {
        if let existing = activePeripheral {
            central.cancelPeripheralConnection(existing)
        }
        activePeripheral = peripheral
        peripheral.delegate = self
        status = .connecting
        central.connect(peripheral, options: nil)
        log("Connecting (delta=\(connectReasonDelta)) to \(peripheral.name ?? "Shadow")")
    }
    
    private func requestMissedEvents() {
        guard let characteristic = eventChar,
              let peripheral = activePeripheral else {
            log("Cannot request missed events: missing characteristic/peripheral")
            status = .error
            return
        }
        status = .requestingMissed
        
        // Write ONE byte: lastKnownSequence (the highest the Mac has)
        var payload = Data(count: 1)
        payload[0] = lastKnownSequence
        log("Writing lastKnownSequence=\(lastKnownSequence) to request missed events")
        peripheral.writeValue(payload, for: characteristic, type: .withResponse)
        
        // After write completes, we read (explicit)
        peripheral.readValue(for: characteristic)
    }
    
    private func processResponse(_ data: Data) {
        // Minimal form (delta <= 1): [currentSeq, currentState] length==2
        if data.count == 2 {
            let seq = data[0]
            let st = data[1]
            log("Minimal response seq=\(seq) state=\(st)")
            lastKnownSequence = seq
            currentStableState = st
            missedEvents.removeAll()
            saveSequence(seq)
            status = .upToDate
            disconnect()
            return
        }
        
        // Extended form: Byte0 seq, Byte1 state, Byte2 missedCount, then pairs
        guard data.count >= 3 else {
            log("Invalid response length=\(data.count)")
            status = .error
            disconnect()
            return
        }
        
        let currentSeq = data[0]
        let currentState = data[1]
        let missedCount = data[2]
        let expectedLen = 3 + Int(missedCount) * 2
        
        if data.count < expectedLen {
            log("Truncated response. Expected \(expectedLen) got \(data.count)")
        }
        
        var events: [(UInt8, UInt8)] = []
        for i in 0..<missedCount {
            let base = 3 + Int(i) * 2
            if base + 1 < data.count {
                let seq = data[base]
                let st = data[base + 1]
                events.append((seq, st))
            }
        }
        
        missedEvents = events
        lastKnownSequence = currentSeq
        currentStableState = currentState
        saveSequence(currentSeq)
        
        let missedList = events.map { String($0.0) }.joined(separator: ",")
        log("Extended response: currentSeq=\(currentSeq) state=\(currentState) missedCount=\(missedCount) missed=[\(missedList)]")
        
        status = .upToDate
        disconnect()
    }
    
    private func disconnect() {
        guard let p = activePeripheral else { return }
        central.cancelPeripheralConnection(p)
    }
    
    // MARK: - Persistence
    private func saveSequence(_ seq: UInt8) {
        UserDefaults.standard.set(Int(seq), forKey: sequenceKey)
    }
    private func loadLastKnownSequence() {
        let stored = UserDefaults.standard.integer(forKey: sequenceKey)
        lastKnownSequence = UInt8(stored & 0x7F)
    }
    
    // MARK: - Logging
    private func log(_ msg: String) {
        let ts = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(ts)] \(msg)"
        print(line)
        logLines.append(line)
        if logLines.count > 200 {
            logLines.removeFirst(logLines.count - 200)
        }
    }
}

// MARK: - CBCentralManagerDelegate
extension LightShadowBLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        if central.state == .poweredOn {
            log("Bluetooth powered ON")
        } else if central.state == .poweredOff {
            log("Bluetooth powered OFF")
            status = .error
        }
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any],
                        rssi RSSI: NSNumber) {
        guard peripheral.name == "Shadow" else { return }
        guard let serviceData = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data],
              let combined = serviceData[serviceUUID] else {
            return
        }
        handleAdvertisement(peripheral: peripheral, serviceData: combined)
    }
    
    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        log("Connected. Discovering services...")
        peripheral.discoverServices([serviceUUID])
    }
    
    func centralManager(_ central: CBCentralManager,
                        didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        log("Failed to connect: \(error?.localizedDescription ?? "unknown")")
        status = .error
        activePeripheral = nil
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        log("Disconnected")
        activePeripheral = nil
        eventChar = nil
        if status != .error {
            status = .scanning
        }
    }
}

// MARK: - CBPeripheralDelegate
extension LightShadowBLEManager: CBPeripheralDelegate {
    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverServices error: Error?) {
        if let error {
            log("Service discovery error: \(error)")
            status = .error
            disconnect()
            return
        }
        guard let services = peripheral.services else {
            log("No services")
            status = .error
            disconnect()
            return
        }
        for s in services where s.uuid == serviceUUID {
            peripheral.discoverCharacteristics([eventCharUUID], for: s)
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let error {
            log("Characteristic discovery error: \(error)")
            status = .error
            disconnect()
            return
        }
        service.characteristics?.forEach { c in
            if c.uuid == eventCharUUID { eventChar = c }
        }
        guard eventChar != nil else {
            log("Event characteristic not found")
            status = .error
            disconnect()
            return
        }
        
        let delta = modularDelta(from: lastKnownSequence, to: advertisedSequence)
        if delta <= 1 && !alwaysConnectOnChange {
            log("Delta <= 1 during connection; reading minimal response")
            peripheral.readValue(for: eventChar!)
        } else {
            requestMissedEvents()
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didWriteValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error {
            log("Write error: \(error.localizedDescription)")
            status = .error
        } else {
            log("Write OK")
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error {
            log("Read error: \(error.localizedDescription)")
            status = .error
            return
        }
        guard let data = characteristic.value else {
            log("No data in update")
            return
        }
        processResponse(data)
    }
}