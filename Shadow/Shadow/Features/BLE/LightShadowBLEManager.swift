@preconcurrency import Foundation
@preconcurrency import CoreBluetooth
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
    private let timeSyncCharUUID = CBUUID(string: "B005")  // Time synchronization
    
    private let ringBufferCapacity: UInt8 = 32
    private let resetOpcode: UInt8 = 0xFF
    private let resetMagic: UInt8 = 0x52
    private let connectThrottle: TimeInterval = 1.5
    private let alwaysConnectOnChange = false  // Only connect when delta > 1
    
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
    private var timeSyncChar: CBCharacteristic?
    private var pendingChallenge: SecurityChallenge?
    
    // Internal
    private var advSeq: UInt8 = 0
    private var advState: UInt8 = 0
    private var delta: UInt8 = 0
    private var lastConnectAttempt = Date.distantPast
    private var pendingReset = false
    private var pendingTimeSync = false  // Whether to sync time after connection
    private var lastTimeSyncDate = Date.distantPast  // Track last successful time sync
    
    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: .main)
        lastKnownSequence = repo.loadLastKnownSequence(deviceUUID: deviceUUID)
        currentStableState = 0
        log("Manager init, lastKnownSequence=\(lastKnownSequence)")
    }
    
    // MARK: Public Interface
    
    /// Check if a Shadow device is paired
    var isPairedToDevice: Bool {
        UserDefaults.standard.string(forKey: "PairedShadowDevice") != nil
    }
    
    /// Get paired device name
    var pairedDeviceName: String? {
        UserDefaults.standard.string(forKey: "PairedShadowDevice")
    }
    
    /// Unpair current device
    func unpairDevice() {
        UserDefaults.standard.removeObject(forKey: "PairedShadowDevice")
        stop()
        log("Device unpaired")
    }
    
    func start() {
        guard central.state == .poweredOn else {
            log("⚠️ Bluetooth not powered on (state: \(central.state.rawValue))")
            return
        }
        
        // Check if device is paired
        guard isPairedToDevice else {
            log("⚠️ No paired device. Please scan QR code first.")
            status = .idle
            return
        }
        
        if isScanning { 
            log("Already scanning, ignoring start() call")
            return
        }
        isScanning = true
        status = .scanning
        central.scanForPeripherals(withServices: nil,
                                   options: [CBCentralManagerScanOptionAllowDuplicatesKey: true])
        log("✅ Started scanning for \(pairedDeviceName ?? "unknown device")...")
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
        guard data.count == 1 else { 
            log("⚠️ Invalid adv data size: \(data.count)")
            return 
        }
        let b = data[0]
        let seq = (b >> 1) & 0x7F
        let st = b & 0x01
        
        let d = modularDelta(old: lastKnownSequence, new: seq)
        
        log("🔍 handleAdv: seq=\(seq) state=\(st) lastKnown=\(lastKnownSequence) delta=\(d)")
        
        // Check for initial state (device just booted/reset)
        let isInitialState = (seq == 0 && st == 0)
        
        // CRITICAL: If device is in initial state (seq=0 state=0), connect and sync time
        // DEVICE RESET DETECTION: If lastKnown > 0 and we see seq=0, device definitely reset!
        let isDeviceReset = (lastKnownSequence > 0 && seq == 0)
        
        if isInitialState {
            // If device reset (was seq>0, now seq=0), ALWAYS reconnect regardless of time
            if isDeviceReset {
                log("🔄 DEVICE RESET DETECTED! (lastKnown=\(lastKnownSequence) -> seq=0) Forcing reconnect...")
                advSeq = seq; advState = st; delta = d
                connect(reset: true, peripheral: peripheral, syncTime: true)
                return
            }
            
            // Otherwise check 5-minute rule (first boot or app restart)
            let timeSinceLastSync = Date().timeIntervalSince(lastTimeSyncDate)
            if timeSinceLastSync < 300 {  // 5 minutes
                log("⏰ Initial state but time synced \(Int(timeSinceLastSync))s ago, skipping reconnect")
                return
            }
            
            log("Initial state detected (seq=0 state=0) -> connect & sync time")
            advSeq = seq; advState = st; delta = d
            connect(reset: false, peripheral: peripheral, syncTime: true)
            return
        }
        
        // Check if state changed even if delta is 0 (device at same sequence but different state)
        if d == 0 {
            if st != currentStableState {
                log("🔄 State changed from \(currentStableState) to \(st) at same sequence \(seq) - applying update!")
                advSeq = seq; advState = st; delta = d
                applySimple(seq: seq, state: st)
                return
            }
            log("⚠️ Delta is 0 and state unchanged (seq=\(seq), state=\(st), lastKnownState=\(currentStableState)), skipping")
            return
        }
        
        advSeq = seq; advState = st; delta = d
        log("ADV seq=\(seq) state=\(st) delta=\(d)")
        
        // Apply locally if delta=1 (no connection needed)
        if d == 1 && !alwaysConnectOnChange {
            applySimple(seq: seq, state: st)
            return
        }
        
        // Connect for delta > 1
        if d > 1 && d <= ringBufferCapacity {
            connect(reset: false, peripheral: peripheral, syncTime: true)
        } else if d > ringBufferCapacity {
            log("Large gap \(d) > \(ringBufferCapacity) -> reset")
            connect(reset: true, peripheral: peripheral, syncTime: true)
        }
    }
    
    private func applySimple(seq: UInt8, state: UInt8) {
        let previousState = currentStableState
        
        log("📝 applySimple: seq=\(seq), state=\(state), previousState=\(previousState)")
        
        lastKnownSequence = seq
        currentStableState = state
        repo.updateDeviceState(deviceUUID: deviceUUID,
                               sequence: seq,
                               state: state,
                               resetCounter: repo.currentResetCounter(deviceUUID: deviceUUID),
                               epoch: nil)
        log("Applied simple delta=1 update locally")
        status = .upToDate
        
        // Always persist for timeline tracking (even if state unchanged)
        persistTransition(seq: seq, st: state, note: previousState != state ? "state-change" : "sequence-update")
        
        // Send notification on stress state change
        if previousState != state {
            Task { @MainActor in
                if state == 1 {
                    // Stress detected
                    NotificationManager.shared.sendStressAlert()
                } else {
                    // Stress recovered
                    NotificationManager.shared.sendStressRecoveryNotification()
                }
            }
        }
    }
    
    // MARK: Connection Flow
    private func connect(reset: Bool, peripheral: CBPeripheral, syncTime: Bool = false) {
        // Don't connect if already connected or connecting
        if self.peripheral != nil && (status == .connecting || status == .requestingMissed) {
            log("Already connected/connecting, ignoring")
            return
        }
        
        guard Date().timeIntervalSince(lastConnectAttempt) > connectThrottle else {
            log("Connect throttled")
            return
        }
        lastConnectAttempt = Date()
        pendingReset = reset
        pendingTimeSync = syncTime  // Store for after connection
        self.peripheral = peripheral
        peripheral.delegate = self
        status = .connecting
        central.connect(peripheral, options: nil)
        log("Connecting reset=\(reset) syncTime=\(syncTime) delta=\(delta)")
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
    
    private func syncTimeWithDevice() {
        guard let char = timeSyncChar, let p = peripheral else {
            log("⚠️ Time sync failed: characteristic or peripheral not available")
            pendingTimeSync = false
            return
        }
        
        // Get current Unix timestamp in milliseconds
        let now = Date()
        let unixTimestampMs = UInt64(now.timeIntervalSince1970 * 1000)
        
        // Get timezone offset in seconds
        let timezoneOffset = Int32(TimeZone.current.secondsFromGMT())
        
        // Build 12-byte payload:
        // Bytes 0-7: Unix timestamp (uint64_t, little-endian)
        // Bytes 8-11: Timezone offset (int32_t, little-endian)
        var data = Data(count: 12)
        withUnsafeBytes(of: unixTimestampMs.littleEndian) { data.replaceSubrange(0..<8, with: $0) }
        withUnsafeBytes(of: timezoneOffset.littleEndian) { data.replaceSubrange(8..<12, with: $0) }
        
        p.writeValue(data, for: char, type: .withResponse)
        
        let timezoneHours = Double(timezoneOffset) / 3600.0
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        log("⏰ Syncing time: \(dateFormatter.string(from: now)) (UTC\(timezoneHours >= 0 ? "+" : "")\(String(format: "%.1f", timezoneHours)))")
        log("   Unix: \(unixTimestampMs) ms, TZ offset: \(timezoneOffset) sec")
        
        // Don't clear pendingTimeSync here - wait for write callback
        // It will be cleared in didWriteValueFor when write completes
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
        
        log("📦 RAW MINIMAL DATA: [\(data.map { String(format: "%02X", $0) }.joined(separator: " "))]")
        log("📊 PARSED: seq=\(seq), state=\(st) (byte[1]=0x\(String(format: "%02X", data[1])))")
        
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
        
        // Don't disconnect if time sync is pending - wait for write to complete
        if !pendingTimeSync {
            disconnect()
        } else {
            log("⏰ Keeping connection open for time sync...")
        }
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
        
        let previousState = currentStableState
        
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
        
        // Send notification on stress state change
        if previousState != curState {
            Task { @MainActor in
                if curState == 1 {
                    // Stress detected
                    NotificationManager.shared.sendStressAlert()
                } else {
                    // Stress recovered
                    NotificationManager.shared.sendStressRecoveryNotification()
                }
            }
        }
    }
    
    private func persistTransition(seq: UInt8,
                                   st: UInt8,
                                   reset: Int32? = nil,
                                   note: String?) {
        let rc = reset ?? repo.currentResetCounter(deviceUUID: deviceUUID)
        
        log("💾 PERSISTING: seq=\(seq), state=\(st), reset=\(rc), note=\(note ?? "none")")
        
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
        
        log("✅ PERSISTED to CoreData: seq=\(seq), state=\(st)")
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
            nonisolated(unsafe) let observer = UnsafeMutablePointer<NSObjectProtocol?>.allocate(capacity: 1)
            observer.initialize(to: nil)
            
            observer.pointee = NotificationCenter.default.addObserver(
                forName: NSNotification.Name("BLE.CharacteristicRead.\(characteristic.uuid.uuidString)"),
                object: nil,
                queue: .main
            ) { notification in
                if let obs = observer.pointee {
                    NotificationCenter.default.removeObserver(obs)
                }
                observer.deinitialize(count: 1)
                observer.deallocate()
                
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
                if let obs = observer.pointee {
                    NotificationCenter.default.removeObserver(obs)
                    observer.deinitialize(count: 1)
                    observer.deallocate()
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
            nonisolated(unsafe) let observer = UnsafeMutablePointer<NSObjectProtocol?>.allocate(capacity: 1)
            observer.initialize(to: nil)
            
            observer.pointee = NotificationCenter.default.addObserver(
                forName: NSNotification.Name("BLE.CharacteristicWrite.\(characteristic.uuid.uuidString)"),
                object: nil,
                queue: .main
            ) { notification in
                if let obs = observer.pointee {
                    NotificationCenter.default.removeObserver(obs)
                }
                observer.deinitialize(count: 1)
                observer.deallocate()
                
                if let error = notification.userInfo?["error"] as? Error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            
            peripheral.writeValue(value, for: characteristic, type: .withResponse)
            
            // Timeout after 5 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if let obs = observer.pointee {
                    NotificationCenter.default.removeObserver(obs)
                    observer.deinitialize(count: 1)
                    observer.deallocate()
                    continuation.resume(throwing: PairingError.timeout)
                }
            }
        }
    }
}

// MARK: - CBCentralManagerDelegate
extension LightShadowBLEManager: CBCentralManagerDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
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
    }
    
    nonisolated func centralManager(_ central: CBCentralManager,
                        didDiscover peripheral: CBPeripheral,
                        advertisementData: [String : Any],
                        rssi RSSI: NSNumber) {
        Task { @MainActor in
            // Filter by paired device name (from QR code scan)
            let pairedDevice = UserDefaults.standard.string(forKey: "PairedShadowDevice")
            
            // If no paired device, ignore all advertisements
            guard let pairedDeviceName = pairedDevice else {
                // DEBUG: Log when no paired device
                if peripheral.name?.hasPrefix("Shadow-") == true {
                    log("🔍 Discovered \(peripheral.name ?? "unknown") but NO paired device in UserDefaults")
                }
                return
            }
            
            // DEBUG: Log all Shadow device discoveries
            if peripheral.name?.hasPrefix("Shadow-") == true {
                log("🔍 Discovered: \(peripheral.name ?? "unknown"), paired: \(pairedDeviceName), match: \(peripheral.name == pairedDeviceName)")
            }
            
            // Only process advertisements from our paired device
            guard peripheral.name == pairedDeviceName else { return }
        
            // Check for service data
            let hasServiceData = advertisementData[CBAdvertisementDataServiceDataKey] != nil
            log("📡 Service data present: \(hasServiceData)")
            
            guard let sd = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data] else {
                log("⚠️ No service data dictionary in advertisement")
                return
            }
            
            log("📡 Service UUIDs in advertisement: \(sd.keys.map { $0.uuidString })")
            
            guard let d = sd[serviceUUID] else {
                log("⚠️ Service UUID \(serviceUUID.uuidString) not found in advertisement")
                return
            }
            
            log("✅ Calling handleAdv with \(d.count) bytes")
            handleAdv(peripheral: peripheral, data: d)
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager,
                        didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            log("Connected -> discover services")
            // Discover both stress service and pairing service
            peripheral.discoverServices([serviceUUID, pairingServiceUUID])
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager,
                        didFailToConnect peripheral: CBPeripheral,
                        error: Error?) {
        Task { @MainActor in
            log("Failed connect: \(error?.localizedDescription ?? "unknown")")
            status = .error
            self.peripheral = nil
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager,
                        didDisconnectPeripheral peripheral: CBPeripheral,
                        error: Error?) {
        Task { @MainActor in
            log("Disconnected (err=\(error?.localizedDescription ?? "none"))")
            eventChar = nil
            self.peripheral = nil
            if status != .error {
                status = .scanning
            }
        }
    }
}

// MARK: - CBPeripheralDelegate
extension LightShadowBLEManager: CBPeripheralDelegate {
    nonisolated func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverServices error: Error?) {
        Task { @MainActor in
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
                    // Discover all pairing characteristics including time sync
                    peripheral.discoverCharacteristics([
                        deviceInfoCharUUID,
                        pairingStateCharUUID,
                        pairingControlCharUUID,
                        securityChallengeCharUUID,
                        timeSyncCharUUID
                    ], for: $0)
                }
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService,
                    error: Error?) {
        Task { @MainActor in
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
                    log("📡 Found Event characteristic (A002)")
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
                else if $0.uuid == timeSyncCharUUID {
                    timeSyncChar = $0
                    log("⏰ Found Time Sync characteristic")
                }
            }
            
            // Sync time if requested AND pairing service is discovered AND time sync char available
            if service.uuid == pairingServiceUUID && pendingTimeSync && timeSyncChar != nil {
                syncTimeWithDevice()
            }
            
            // Original stress service logic - only proceed if stress service is discovered
            if service.uuid == serviceUUID {
                guard eventChar != nil else {
                    log("Event characteristic missing from stress service")
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
            // For pairing service, just log that it's ready
            if service.uuid == pairingServiceUUID {
                log("✅ Pairing service characteristics discovered")
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral,
                    didWriteValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        Task { @MainActor in
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
                
                // If this was a time sync write, mark success and disconnect
                if characteristic.uuid == timeSyncCharUUID {
                    pendingTimeSync = false  // Clear pending flag
                    lastTimeSyncDate = Date()  // Track successful sync
                    log("⏰ Time sync write complete, disconnecting...")
                    disconnect()
                }
                
                // Post notification for async write success
                NotificationCenter.default.post(
                    name: NSNotification.Name("BLE.CharacteristicWrite.\(characteristic.uuid.uuidString)"),
                    object: nil
                )
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic,
                    error: Error?) {
        Task { @MainActor in
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
}
