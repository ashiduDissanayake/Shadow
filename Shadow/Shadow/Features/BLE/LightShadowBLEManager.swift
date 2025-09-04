import Foundation
import CoreBluetooth
import Combine

// LightShadowBLEManager
// Simplified advertisement-based delta detection with missed-event replay.
// Now supports "flush on large gap": if delta > ringBufferCapacity, we connect
// and send RESET opcode (0xFF). Device flushes its event log and resets sequence to 0.
// Replay Protocol:
//   - Normal write: 1 byte (clientLastKnownSequence) => read response
//   - Reset write:  0xFF => read reset ACK response
//
// Responses:
//   Minimal (2 bytes): [currentSequence, currentState]
//   Extended:          [currentSequence, currentState, missedCount, (seq,state)*N]
//   Reset ACK (4 b):   [0x00, stateBit, 0x00, 0x52] (magic 'R'=0x52)
//
// Sequence: 7-bit (0..127) rolling, modular arithmetic.

final class LightShadowBLEManager: NSObject, ObservableObject {
    
    // MARK: - Published (UI)
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
    
    private let alwaysConnectOnChange = false
    
    // Flush-on-large-gap strategy parameters
    private let ringBufferCapacity: UInt8 = 32
    private let flushOnLargeGap = true
    private let resetOpcode: UInt8 = 0xFF
    private let resetMagic: UInt8 = 0x52
    
    // MARK: - CoreBluetooth
    private var central: CBCentralManager!
    private var activePeripheral: CBPeripheral?
    private var eventChar: CBCharacteristic?
    
    // MARK: - Internal state
    private var advertisedSequence: UInt8 = 0
    private var advertisedState: UInt8 = 0
    private var connectReasonDelta: UInt8 = 0
    private var lastConnectAttempt = Date.distantPast
    private let connectThrottle: TimeInterval = 1.5
    
    private var pendingResetAfterConnect = false
    
    // Persistence key
    private let sequenceKey = "Shadow_LastKnownSequence_V1"
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
        loadLastKnownSequence()
        log("Initialized. lastKnownSequence=\(lastKnownSequence)")
    }
    
    // MARK: - Control
    func start() {
        guard central.state == .poweredOn else {
            log("Bluetooth not powered on.")
            return
        }
        if isScanning { return }
        isScanning = true
        status = .scanning
        central.scanForPeripherals(withServices: nil,
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
        log("Scanning started (duplicates allowed)")
    }
    
    func stop() {
        guard isScanning else { return }
        central.stopScan()
        isScanning = false
        status = .idle
        log("Scanning stopped")
    }
    
    // MARK: - Advertisement Handling
    private func modularDelta(from old: UInt8, to new: UInt8) -> UInt8 {
        (new &- old) & 0x7F
    }
    
    private func handleAdvertisement(peripheral: CBPeripheral,
                                     serviceData: Data) {
        guard serviceData.count == 1 else { return }
        let combined = serviceData[0]
        let seq = (combined >> 1) & 0x7F
        let state = combined & 0x01
        
        let delta = modularDelta(from: lastKnownSequence, to: seq)
        guard delta != 0 else { return } // no change
        
        advertisedSequence = seq
        advertisedState = state
        connectReasonDelta = delta
        
        log("Advertisement change: advSeq=\(seq) state=\(state) delta=\(delta)")
        
        // Path 1: simple local update
        if delta == 1 && !alwaysConnectOnChange {
            applySimpleUpdate(sequence: seq, state: state)
            return
        }
        
        // Path 2: delta within missed replay capacity (2..32)
        if delta <= ringBufferCapacity {
            attemptConnection(forReset: false, peripheral: peripheral)
            return
        }
        
        // Path 3: Large gap => flush (if enabled) or attempt partial replay
        if flushOnLargeGap {
            log("⚠️ Large delta (\(delta)) > capacity (\(ringBufferCapacity)). Initiating reset flush.")
            attemptConnection(forReset: true, peripheral: peripheral)
        } else {
            log("⚠️ Large delta but flush disabled; attempting partial fetch anyway.")
            attemptConnection(forReset: false, peripheral: peripheral)
        }
    }
    
    private func attemptConnection(forReset: Bool, peripheral: CBPeripheral) {
        guard Date().timeIntervalSince(lastConnectAttempt) > connectThrottle else {
            log("Connection throttled; skipping.")
            return
        }
        pendingResetAfterConnect = forReset
        lastConnectAttempt = Date()
        connectTo(peripheral)
    }
    
    private func applySimpleUpdate(sequence: UInt8, state: UInt8) {
        lastKnownSequence = sequence
        currentStableState = state
        missedEvents.removeAll()
        saveSequence(sequence)
        status = .upToDate
        log("Applied simple update locally seq=\(sequence) state=\(state)")
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
        log("Connecting (delta=\(connectReasonDelta)) reset=\(pendingResetAfterConnect)")
    }
    
    private func requestMissedEvents(from clientLastKnown: UInt8) {
        guard let characteristic = eventChar,
              let peripheral = activePeripheral else {
            log("Missing characteristic/peripheral for missed events request")
            status = .error
            return
        }
        status = .requestingMissed
        
        var payload = Data(count: 1)
        payload[0] = clientLastKnown
        log("Requesting missed events with lastKnownSequence=\(clientLastKnown)")
        peripheral.writeValue(payload, for: characteristic, type: .withResponse)
        peripheral.readValue(for: characteristic)
    }
    
    private func sendResetOpcode() {
        guard let characteristic = eventChar,
              let peripheral = activePeripheral else {
            log("Missing characteristic/peripheral for reset")
            status = .error
            return
        }
        status = .requestingMissed
        
        var payload = Data(count: 1)
        payload[0] = resetOpcode
        log("Writing RESET opcode (0xFF)")
        peripheral.writeValue(payload, for: characteristic, type: .withResponse)
        peripheral.readValue(for: characteristic)
    }
    
    private func disconnect() {
        guard let p = activePeripheral else { return }
        central.cancelPeripheralConnection(p)
    }
    
    // MARK: - Response Parsing
    private func handleResetAck(data: Data) -> Bool {
        // Expect: [0x00, stateBit, 0x00, 0x52]
        guard data.count >= 4,
              data[0] == 0x00,
              data[2] == 0x00,
              data[3] == resetMagic else {
            return false
        }
        let st = data[1] & 0x01
        lastKnownSequence = 0
        currentStableState = st
        saveSequence(0)
        missedEvents.removeAll()
        log("✅ Reset ACK: sequence reset to 0 (state=\(st))")
        pendingResetAfterConnect = false
        status = .upToDate
        return true
    }
    
    private func processMissedEventsResponse(_ data: Data) {
        // Extended response:
        // Byte0: currentSequence
        // Byte1: currentStateBit
        // Byte2: missedCount
        // Then pairs: (seq_i, state_i)
        
        guard data.count >= 3 else {
            log("Invalid extended response length=\(data.count)")
            status = .error
            disconnect()
            return
        }
        let currentSeq = data[0]
        let currentState = data[1]
        let missedCount = data[2]
        let expected = 3 + Int(missedCount) * 2
        
        if data.count < expected {
            log("Truncated extended response. expected=\(expected) got=\(data.count)")
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
        
        let missedList = events.map { "\($0.0)" }.joined(separator: ",")
        log("Extended: currentSeq=\(currentSeq) state=\(currentState) missedCount=\(missedCount) missed=[\(missedList)]")
        status = .upToDate
        disconnect()
    }
    
    private func processMinimalResponse(_ data: Data) {
        // Minimal is 2 bytes
        guard data.count == 2 else {
            log("Unexpected minimal response length=\(data.count)")
            return
        }
        let seq = data[0]
        let st = data[1]
        lastKnownSequence = seq
        currentStableState = st
        saveSequence(seq)
        missedEvents.removeAll()
        log("Minimal response seq=\(seq) state=\(st)")
        status = .upToDate
        disconnect()
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
        if logLines.count > 300 {
            logLines.removeFirst(logLines.count - 300)
        }
    }
}

// MARK: - CBCentralManagerDelegate
extension LightShadowBLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            log("Bluetooth powered ON")
        case .poweredOff:
            log("Bluetooth powered OFF")
            status = .error
        default:
            break
        }
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any],
                        rssi RSSI: NSNumber) {
        guard peripheral.name == "Shadow" else { return }
        guard let serviceDataMap = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data],
              let data = serviceDataMap[serviceUUID] else { return }
        handleAdvertisement(peripheral: peripheral, serviceData: data)
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
        log("Disconnected (error=\(error?.localizedDescription ?? "none"))")
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
            log("No services found")
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
        guard let chars = service.characteristics else {
            log("No characteristics")
            status = .error
            disconnect()
            return
        }
        for c in chars where c.uuid == eventCharUUID {
            eventChar = c
        }
        guard eventChar != nil else {
            log("Event characteristic not found")
            status = .error
            disconnect()
            return
        }
        
        // Decide path
        if pendingResetAfterConnect {
            sendResetOpcode()
            return
        }
        
        let delta = modularDelta(from: lastKnownSequence, to: advertisedSequence)
        if delta <= 1 && !alwaysConnectOnChange {
            // Minimal
            peripheral.readValue(for: eventChar!)
        } else {
            // Missed events
            requestMissedEvents(from: lastKnownSequence)
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
        
        // Reset ACK?
        if pendingResetAfterConnect {
            if handleResetAck(data: data) {
                disconnect()
                return
            } else {
                log("Unexpected reset ACK format (len=\(data.count)); continuing parse")
                pendingResetAfterConnect = false
            }
        }
        
        // Minimal vs extended
        switch data.count {
        case 2:
            processMinimalResponse(data)
        case 3...:
            processMissedEventsResponse(data)
        default:
            log("Unhandled response length=\(data.count)")
        }
    }
}
