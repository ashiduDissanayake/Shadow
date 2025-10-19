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
    
    // Pairing-related published properties
    @Published var isPaired: Bool = false
    @Published var pairingState: PairingState = .idle
    @Published var deviceInfo: DeviceInfo?
    
    // MARK: Config
    private let serviceUUID = CBUUID(string: "A000")
    private let eventCharUUID = CBUUID(string: "A002")
    
    // Pairing Service UUIDs
    private let pairingServiceUUID = CBUUID(string: "B000")
    private let deviceInfoCharUUID = CBUUID(string: "B001")
    private let pairingStateCharUUID = CBUUID(string: "B002")
    private let pairingControlCharUUID = CBUUID(string: "B003")
    private let securityChallengeCharUUID = CBUUID(string: "B004")
    
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
    
    // Pairing characteristics
    private var deviceInfoChar: CBCharacteristic?
    private var pairingStateChar: CBCharacteristic?
    private var pairingControlChar: CBCharacteristic?
    private var securityChallengeChar: CBCharacteristic?
    private var pendingChallenge: SecurityChallenge?
    
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
    
    // MARK: - Pairing Methods
    
    /// Perform pairing with Shadow device
    func performPairing() async throws {
        guard let peripheral = peripheral else {
            throw PairingError.deviceNotConnected
        }
        
        log("🔐 Starting pairing process...")
        
        // Step 1: Read device info
        guard let deviceInfoChar = deviceInfoChar else {
            throw PairingError.characteristicNotFound
        }
        
        let deviceInfoData = try await readCharacteristic(deviceInfoChar, from: peripheral)
        guard let info = DeviceInfo(from: deviceInfoData) else {
            throw PairingError.invalidData
        }
        self.deviceInfo = info
        
        log("📱 Shadow Device: \(info.deviceName)")
        log("🆔 Device ID: \(info.deviceIDHex)")
        log("🔧 Firmware: \(info.firmwareVersion)")
        log("⚙️ Hardware: \(info.hardwareRevision)")
        
        // Step 2: Send pair request
        guard let pairingControlChar = pairingControlChar else {
            throw PairingError.characteristicNotFound
        }
        
        let pairCommand = Data([PairingCommand.pairRequest.rawValue])
        try await writeCharacteristic(pairingControlChar, value: pairCommand, to: peripheral)
        log("📤 Sent pairing request")
        
        // Step 3: Wait for pairing state to change to PENDING
        try await waitForPairingState(.pending, timeout: 5.0)
        log("⏳ Pairing state: PENDING")
        
        // Step 4: Read challenge
        guard let securityChallengeChar = securityChallengeChar else {
            throw PairingError.characteristicNotFound
        }
        
        let challengeData = try await readCharacteristic(securityChallengeChar, from: peripheral)
        guard let challenge = SecurityChallenge(from: challengeData) else {
            throw PairingError.invalidData
        }
        self.pendingChallenge = challenge
        
        log("🔐 Received challenge (timestamp: \(challenge.timestamp))")
        
        // Step 5: Compute response
        let clientDeviceID = PairingHelper.getOrCreateClientDeviceID()
        let clientName = PairingHelper.getClientDeviceName()
        
        let responseData = PairingHelper.prepareChallengeResponse(
            challenge: challenge.challenge,
            shadowDeviceID: info.deviceID,
            clientDeviceID: clientDeviceID,
            clientName: clientName
        )
        
        // Step 6: Send response
        try await writeCharacteristic(securityChallengeChar, value: responseData, to: peripheral)
        log("📤 Sent challenge response")
        
        // Step 7: Wait for pairing state to change to PAIRED
        try await waitForPairingState(.paired, timeout: 5.0)
        
        log("✅ Pairing successful!")
        isPaired = true
        
        // Save pairing info
        PairingHelper.savePairingInfo(deviceInfo: info, clientDeviceID: clientDeviceID)
    }
    
    /// Wait for pairing state to change to target state
    private func waitForPairingState(_ targetState: PairingState, timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        
        while Date() < deadline {
            if pairingState == targetState {
                return
            }
            
            if pairingState == .rejected {
                throw PairingError.rejected
            }
            
            try await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        }
        
        throw PairingError.timeout
    }
    
    /// Read characteristic value (async wrapper)
    private func readCharacteristic(_ characteristic: CBCharacteristic, 
                                   from peripheral: CBPeripheral) async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            var observer: NSObjectProtocol?
            
            observer = NotificationCenter.default.addObserver(
                forName: NSNotification.Name("BLE.CharacteristicRead.\(characteristic.uuid.uuidString)"),
                object: nil,
                queue: .main
            ) { notification in
                if let observer = observer {
                    NotificationCenter.default.removeObserver(observer)
                }
                
                if let error = notification.userInfo?["error"] as? Error {
                    continuation.resume(throwing: error)
                } else if let data = notification.userInfo?["data"] as? Data {
                    continuation.resume(returning: data)
                } else {
                    continuation.resume(throwing: PairingError.invalidData)
                }
            }
            
            peripheral.readValue(for: characteristic)
            
            // Timeout after 5 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if let observer = observer {
                    NotificationCenter.default.removeObserver(observer)
                    continuation.resume(throwing: PairingError.timeout)
                }
            }
        }
    }
    
    /// Write characteristic value (async wrapper)
    private func writeCharacteristic(_ characteristic: CBCharacteristic,
                                    value: Data,
                                    to peripheral: CBPeripheral) async throws {
        return try await withCheckedThrowingContinuation { continuation in
            var observer: NSObjectProtocol?
            
            observer = NotificationCenter.default.addObserver(
                forName: NSNotification.Name("BLE.CharacteristicWrite.\(characteristic.uuid.uuidString)"),
                object: nil,
                queue: .main
            ) { notification in
                if let observer = observer {
                    NotificationCenter.default.removeObserver(observer)
                }
                
                if let error = notification.userInfo?["error"] as? Error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            
            peripheral.writeValue(value, for: characteristic, type: .withResponse)
            
            // Timeout after 5 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if let observer = observer {
                    NotificationCenter.default.removeObserver(observer)
                    continuation.resume(throwing: PairingError.timeout)
                }
            }
        }
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
            } else if $0.uuid == pairingServiceUUID {
                // Discover all pairing characteristics
                peripheral.discoverCharacteristics([
                    deviceInfoCharUUID,
                    pairingStateCharUUID,
                    pairingControlCharUUID,
                    securityChallengeCharUUID
                ], for: $0)
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
            // Stress service characteristics
            if $0.uuid == eventCharUUID {
                eventChar = $0
            }
            // Pairing service characteristics
            else if $0.uuid == deviceInfoCharUUID {
                deviceInfoChar = $0
                log("📋 Found Device Info characteristic")
            }
            else if $0.uuid == pairingStateCharUUID {
                pairingStateChar = $0
                // Subscribe to pairing state notifications
                peripheral.setNotifyValue(true, for: $0)
                log("🔔 Subscribed to Pairing State notifications")
            }
            else if $0.uuid == pairingControlCharUUID {
                pairingControlChar = $0
                log("📋 Found Pairing Control characteristic")
            }
            else if $0.uuid == securityChallengeCharUUID {
                securityChallengeChar = $0
                log("🔐 Found Security Challenge characteristic")
            }
        }
        
        // Original stress service logic
        guard eventChar != nil else {
            // If pairing characteristics found, that's okay - we're in pairing mode
            if deviceInfoChar != nil {
                log("Pairing service discovered, stress service not required yet")
                return
            }
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
            // Post notification for async write failure
            NotificationCenter.default.post(
                name: NSNotification.Name("BLE.CharacteristicWrite.\(characteristic.uuid.uuidString)"),
                object: nil,
                userInfo: ["error": error]
            )
        } else {
            log("Write OK")
            // Post notification for async write success
            NotificationCenter.default.post(
                name: NSNotification.Name("BLE.CharacteristicWrite.\(characteristic.uuid.uuidString)"),
                object: nil
            )
        }
    }
    
    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        if let error {
            log("Read error: \(error.localizedDescription)")
            status = .error
            
            // Post notification for async read failure
            NotificationCenter.default.post(
                name: NSNotification.Name("BLE.CharacteristicRead.\(characteristic.uuid.uuidString)"),
                object: nil,
                userInfo: ["error": error]
            )
            return
        }
        guard let data = characteristic.value else { return }
        
        // Handle pairing characteristics
        if characteristic.uuid == pairingStateCharUUID {
            if let stateInfo = PairingStateInfo(from: data) {
                pairingState = stateInfo.state
                log("🔐 Pairing state: \(stateInfo.state) (\(stateInfo.pairedCount)/\(stateInfo.maxPaired) paired)")
            }
            // Post notification for async read success
            NotificationCenter.default.post(
                name: NSNotification.Name("BLE.CharacteristicRead.\(characteristic.uuid.uuidString)"),
                object: nil,
                userInfo: ["data": data]
            )
            return
        }
        
        // Post notification for async reads (device info, challenge, etc.)
        NotificationCenter.default.post(
            name: NSNotification.Name("BLE.CharacteristicRead.\(characteristic.uuid.uuidString)"),
            object: nil,
            userInfo: ["data": data]
        )
        
        // Original stress service logic continues below
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
