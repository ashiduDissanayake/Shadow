import Foundation
import CoreBluetooth
import Combine

/// BLE manager aligned to the ViewModel expectations:
/// Published: status, isScanning, lastKnownSequence, currentStableState, logLines
/// Methods: start(), stop()
@MainActor
final class LightShadowBLEManager: NSObject, ObservableObject {
    
    enum Status: String {
        case idle, scanning, connecting, requestingMissed, upToDate, error
    }
    
    // MARK: Published for UI
    @Published var status: Status = .idle
    @Published var isScanning: Bool = false
    @Published var lastKnownSequence: UInt8 = 0
    @Published var currentStableState: UInt8 = 0   // 0=CALM 1=STRESS
    @Published var logLines: [String] = []
    
    // MARK: Config
    private let serviceUUID = CBUUID(string: "A000")
    private let eventCharUUID = CBUUID(string: "A002")
    private let ringBufferCapacity: UInt8 = 32
    private let resetOpcode: UInt8 = 0xFF
    private let resetMagic: UInt8 = 0x52
    private let connectThrottle: TimeInterval = 1.5
    private let alwaysConnectOnChange = true
    
    // Persistence / repository
    private let repo = StressDataRepository.shared
    private var deviceUUID: UUID { repo.defaultDeviceUUID }
    
    // BLE
    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private var eventChar: CBCharacteristic?
    
    // Internal
    private var advSeq: UInt8 = 0
    private var advState: UInt8 = 0
    private var delta: UInt8 = 0
    private var lastConnectAttempt = Date.distantPast
    private var pendingReset = false
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
        lastKnownSequence = repo.loadLastKnownSequence(deviceUUID: deviceUUID)
        currentStableState = 0
        log("Manager init, lastKnownSequence=\(lastKnownSequence)")
    }
    
    // MARK: Public Interface
    func start() {
        guard central.state == .poweredOn else {
            log("Bluetooth not powered on")
            return
        }
        if isScanning { return }
        isScanning = true
        status = .scanning
        central.scanForPeripherals(withServices: nil,
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
        log("Scanning...")
    }
    
    func stop() {
        guard isScanning else { return }
        central.stopScan()
        isScanning = false
        status = .idle
        log("Stopped scanning")
    }
    
    // MARK: Advertisement
    private func modularDelta(old: UInt8, new: UInt8) -> UInt8 {
        (new &- old) & 0x7F
    }
    
    private func handleAdv(peripheral: CBPeripheral, data: Data) {
        guard data.count == 1 else { return }
        let b = data[0]
        let seq = (b >> 1) & 0x7F
        let st = b & 0x01
        
        let d = modularDelta(old: lastKnownSequence, new: seq)
        guard d != 0 else { return }
        
        advSeq = seq; advState = st; delta = d
        log("ADV seq=\(seq) state=\(st) delta=\(d)")
        
        if d == 1 && !alwaysConnectOnChange {
            applySimple(seq: seq, state: st)
            return
        }
        if d <= ringBufferCapacity {
            connect(reset: false, peripheral: peripheral)
        } else {
            log("Large gap \(d) > \(ringBufferCapacity) -> reset")
            connect(reset: true, peripheral: peripheral)
        }
    }
    
    private func applySimple(seq: UInt8, state: UInt8) {
        lastKnownSequence = seq
        currentStableState = state
        repo.updateDeviceState(deviceUUID: deviceUUID,
                               sequence: seq,
                               state: state,
                               resetCounter: repo.currentResetCounter(deviceUUID: deviceUUID),
                               epoch: nil)
        log("Applied simple delta=1 update locally")
        status = .upToDate
    }
    
    // MARK: Connection Flow
    private func connect(reset: Bool, peripheral: CBPeripheral) {
        guard Date().timeIntervalSince(lastConnectAttempt) > connectThrottle else {
            log("Connect throttled")
            return
        }
        lastConnectAttempt = Date()
        pendingReset = reset
        self.peripheral = peripheral
        peripheral.delegate = self
        status = .connecting
        central.connect(peripheral, options: nil)
        log("Connecting reset=\(reset) delta=\(delta)")
    }
    
    private func sendReset() {
        guard let char = eventChar, let p = peripheral else { return }
        var d = Data(count: 1)
        d[0] = resetOpcode
        status = .requestingMissed
        p.writeValue(d, for: char, type: .withResponse)
        p.readValue(for: char)
        log("Sent RESET opcode")
    }
    
    private func requestMissed() {
        guard let char = eventChar, let p = peripheral else { return }
        var d = Data(count: 1)
        d[0] = lastKnownSequence
        status = .requestingMissed
        p.writeValue(d, for: char, type: .withResponse)
        p.readValue(for: char)
        log("Requested missed from lastSeq=\(lastKnownSequence)")
    }
    
    private func disconnect() {
        if let p = peripheral {
            central.cancelPeripheralConnection(p)
        }
    }
    
    // MARK: Parsing
    private func handleResetAck(_ data: Data) -> Bool {
        guard data.count >= 4,
              data[0] == 0x00,
              data[2] == 0x00,
              data[3] == resetMagic else { return false }
        
        let st = data[1] & 0x01
        let newReset = repo.incrementResetCounter(deviceUUID: deviceUUID)
        
        let marker = ResetMarkerDomainEvent(deviceID: deviceUUID,
                                            resetCounter: newReset,
                                            epoch: nil,
                                            reason: "Flush after gap \(delta)",
                                            receivedAt: Date())
        let lastState = Int16(currentStableState)
        repo.persistResetMarker(marker, lastKnownState: lastState)
        
        lastKnownSequence = 0
        currentStableState = st
        repo.updateDeviceState(deviceUUID: deviceUUID,
                               sequence: 0,
                               state: st,
                               resetCounter: newReset,
                               epoch: nil)
        log("Reset ACK: resetCounter=\(newReset)")
        status = .upToDate
        pendingReset = false
        return true
    }
    
    private func parseMinimal(_ data: Data) {
        guard data.count == 2 else { return }
        let seq = data[0]; let st = data[1] & 0x01
        persistTransition(seq: seq, st: st, note: "minimal")
        lastKnownSequence = seq
        currentStableState = st
        repo.updateDeviceState(deviceUUID: deviceUUID,
                               sequence: seq,
                               state: st,
                               resetCounter: repo.currentResetCounter(deviceUUID: deviceUUID),
                               epoch: nil)
        log("Minimal resp seq=\(seq) state=\(st)")
        status = .upToDate
        disconnect()
    }
    
    private func parseExtended(_ data: Data) {
        guard data.count >= 3 else {
            log("Extended too short")
            status = .error
            return
        }
        let curSeq = data[0]
        let curState = data[1] & 0x01
        let missed = data[2]
        let needed = 3 + Int(missed) * 2
        if data.count < needed { log("Truncated extended resp expected=\(needed) got=\(data.count)") }
        let rc = repo.currentResetCounter(deviceUUID: deviceUUID)
        
        if missed > 0 {
            for i in 0..<missed {
                let base = 3 + Int(i) * 2
                if base + 1 < data.count {
                    let seq = data[base]
                    let st = data[base + 1] & 0x01
                    persistTransition(seq: seq, st: st, reset: rc, note: "missed")
                }
            }
        }
        persistTransition(seq: curSeq, st: curState, reset: rc, note: "current")
        
        lastKnownSequence = curSeq
        currentStableState = curState
        repo.updateDeviceState(deviceUUID: deviceUUID,
                               sequence: curSeq,
                               state: curState,
                               resetCounter: rc,
                               epoch: nil)
        log("Extended current=\(curSeq) missed=\(missed)")
        status = .upToDate
        disconnect()
    }
    
    private func persistTransition(seq: UInt8,
                                   st: UInt8,
                                   reset: Int32? = nil,
                                   note: String?) {
        let rc = reset ?? repo.currentResetCounter(deviceUUID: deviceUUID)
        let evt = StressTransitionDomainEvent(
            deviceID: deviceUUID,
            sequence7: seq,
            fullSequence: nil,
            resetCounter: rc,
            epoch: nil,
            stressState: st,
            receivedAt: Date(),
            deviceTimestampMs: nil,
            confidence: nil,
            batteryMv: nil,
            sensorQuality: nil,
            durationPrevMs: nil,
            notes: note,
            type: .transition,
            isSynthetic: false
        )
        repo.persistTransition(evt)
    }
    
    // MARK: Logging
    private func log(_ msg: String) {
        let ts = DateFormatter.localizedString(from: Date(),
                                               dateStyle: .none,
                                               timeStyle: .medium)
        let line = "[\(ts)] \(msg)"
        print(line)
        logLines.append(line)
        if logLines.count > 500 { logLines.removeFirst(logLines.count - 500) }
    }
}

// MARK: - CBCentralManagerDelegate
extension LightShadowBLEManager: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            log("Bluetooth ON")
        case .poweredOff:
            log("Bluetooth OFF")
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
        guard let sd = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data],
              let d = sd[serviceUUID] else { return }
        handleAdv(peripheral: peripheral, data: d)
    }
    
    func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        log("Connected -> discover services")
        peripheral.discoverServices([serviceUUID])
    }
    
    func centralManager(_ central: CBCentralManager,
                        didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        log("Failed connect: \(error?.localizedDescription ?? "unknown")")
        status = .error
        self.peripheral = nil
    }
    
    func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        log("Disconnected (err=\(error?.localizedDescription ?? "none"))")
        eventChar = nil
        self.peripheral = nil
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
        peripheral.services?.forEach {
            if $0.uuid == serviceUUID {
                peripheral.discoverCharacteristics([eventCharUUID], for: $0)
            }
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        if let error {
            log("Char discovery error: \(error)")
            status = .error
            disconnect()
            return
        }
        service.characteristics?.forEach {
            if $0.uuid == eventCharUUID {
                eventChar = $0
            }
        }
        guard eventChar != nil else {
            log("Event characteristic missing")
            status = .error
            disconnect()
            return
        }
        if pendingReset {
            sendReset()
        } else {
            requestMissed()
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didWriteValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error {
            log("Write error: \(error.localizedDescription)")
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
        guard let data = characteristic.value else { return }
        
        if pendingReset {
            if handleResetAck(data) {
                disconnect()
                return
            } else {
                log("Unexpected reset ACK format, continuing parse")
                pendingReset = false
            }
        }
        switch data.count {
        case 2: parseMinimal(data)
        case 3...: parseExtended(data)
        default:
            log("Unknown response len=\(data.count)")
        }
    }
}
