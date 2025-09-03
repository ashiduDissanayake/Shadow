import Foundation
import CoreBluetooth
import Combine
import CoreData

// MARK: - Supporting Models

struct DebugLogEntry: Identifiable {
    let id = UUID()
    let timestamp: Date
    let message: String
    
    var formattedMessage: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return "[\(formatter.string(from: timestamp))] \(message)"
    }
}

// MARK: - Sync Status Enum (Power-Efficient Protocol)
enum SyncStatus: Int, CaseIterable {
    case scanning = 0
    case connecting = 1
    case synchronizing = 2
    case disconnected = 3
    
    var displayName: String {
        switch self {
        case .scanning:
            return "Scanning for Changes"
        case .connecting:
            return "Connecting to Device"
        case .synchronizing:
            return "Syncing Data"
        case .disconnected:
            return "Disconnected"
        }
    }
    
    var shortName: String {
        switch self {
        case .scanning:
            return "Scanning"
        case .connecting:
            return "Connecting"
        case .synchronizing:
            return "Syncing"
        case .disconnected:
            return "Disconnected"
        }
    }
}

// MARK: - Control Commands for ESP32
enum ControlCommand: UInt8 {
    case replayFromSequence = 0x01
    case acknowledgeTransition = 0x04
}

// MARK: - System Status for UI
enum SystemStatus: Int, CaseIterable {
    case scanning = 0
    case connecting = 1
    case synchronizing = 2
    case disconnected = 3
    
    var displayName: String {
        switch self {
        case .scanning:
            return "Looking for Shadow device..."
        case .connecting:
            return "Connecting to Shadow..."
        case .synchronizing:
            return "Downloading stress data..."
        case .disconnected:
            return "Shadow disconnected"
        }
    }
}

// MARK: - Discovered Device Model
struct DiscoveredShadowDevice: Identifiable {
    let id = UUID()
    let peripheral: CBPeripheral
    let name: String
    let rssi: Int
    let advertisedState: SyncStatus
    let advertisedSequence: UInt8
    let lastSeen: Date
    
    init(peripheral: CBPeripheral, rssi: Int, advertisedState: SyncStatus, advertisedSequence: UInt8) {
        self.peripheral = peripheral
        self.name = peripheral.name ?? "Unknown Shadow"
        self.rssi = rssi
        self.advertisedState = advertisedState
        self.advertisedSequence = advertisedSequence
        self.lastSeen = Date()
    }
}

final class ShadowBLEManager: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var isBluetoothPoweredOn = false
    @Published var isScanning = false
    @Published var connectedDevice: ShadowPeripheral?
    @Published var foundShadowDevices: [DiscoveredShadowDevice] = []
    @Published var currentSyncStatus: SyncStatus = .disconnected
    @Published var currentSystemStatus: SystemStatus = .disconnected
    @Published var connectionStatus: String = "Disconnected"
    @Published var debugLog: [DebugLogEntry] = []
    @Published var lastStressEvent: String = "No recent events"
    @Published var lastSequenceNumber: UInt8 = 0
    @Published var pendingSequenceNumber: UInt8?
    @Published var totalEventsReceived: Int = 0
    
    // MARK: - BLE Properties
    private var centralManager: CBCentralManager!
    private var activePeripheral: CBPeripheral?
    private var stressServiceUUID = CBUUID(string: "1800")
    private var fsmCharacteristicUUID = CBUUID(string: "1801")
    private var eventCharacteristicUUID = CBUUID(string: "1802")
    private var controlPointCharacteristicUUID = CBUUID(string: "1804")
    private var ackCharacteristicUUID = CBUUID(string: "1803")
    
    // MARK: - Characteristic References
    private var fsmCharacteristic: CBCharacteristic?
    private var eventCharacteristic: CBCharacteristic?
    private var controlPointCharacteristic: CBCharacteristic?
    private var ackCharacteristic: CBCharacteristic?
    
    // MARK: - State Management
    private var serviceDiscoveryRetryCount = 0
    private let maxServiceDiscoveryRetries = 3
    private var coreDataManager: CoreDataManager
    
    // MARK: - Initialization
    init(coreDataManager: CoreDataManager) {
        self.coreDataManager = coreDataManager
        super.init()
        centralManager = CBCentralManager(delegate: self, queue: .main)
    }
    
    // Convenience initializer for cases where Core Data isn't ready yet
    convenience override init() {
        // Create a placeholder core data manager
        let container = NSPersistentContainer(name: "AppModel")
        let placeholderManager = ShadowCoreDataManager(persistentContainer: container)
        self.init(coreDataManager: placeholderManager)
    }
    
    // Method to update the core data manager once it's available
    func setCoreDataManager(_ manager: CoreDataManager) {
        self.coreDataManager = manager
        addDebugLog("✅ Core Data manager updated")
    }
    
    // MARK: - Core Data Integration Functions
    private func loadLastKnownSequence(for deviceId: UUID) -> UInt8 {
        // Try to get the last known sequence from Core Data
        guard let shadowCoreDataManager = coreDataManager as? ShadowCoreDataManager else {
            addDebugLog("⚠️ Core Data manager not available, using fallback")
            return lastSequenceNumber
        }
        
        // For now, use device identifier as string. In a full implementation,
        // we'd need to get the current user profile and use proper device lookup
        let deviceIdentifier = deviceId.uuidString
        
        // Since we don't have user profile context here, we'll implement a simplified lookup
        // that searches for the device by identifier across all users
        let request = NSFetchRequest<ShadowDevice>(entityName: "ShadowDevice")
        request.predicate = NSPredicate(format: "deviceIdentifier == %@", deviceIdentifier)
        request.fetchLimit = 1
        
        do {
            let devices = try shadowCoreDataManager.context.fetch(request)
            if let device = devices.first {
                let sequence = UInt8(device.lastKnownSequence)
                addDebugLog("📊 Loaded last known sequence \(sequence) for device \(deviceIdentifier)")
                return sequence
            }
        } catch {
            addDebugLog("❌ Core Data error loading sequence: \(error)")
        }
        
        addDebugLog("📊 No stored sequence found for device, starting from 0")
        return 0
    }
    
    private func loadLastKnownEventSequence() -> UInt8 {
        // Get the highest sequence number from all StressEvent records
        // This tells us the last event sequence we successfully processed
        guard let shadowCoreDataManager = coreDataManager as? ShadowCoreDataManager else {
            addDebugLog("⚠️ Core Data manager not available for event sequence lookup")
            return 0
        }
        
        let request = NSFetchRequest<StressEvent>(entityName: "StressEvent")
        request.sortDescriptors = [NSSortDescriptor(key: "sequenceNumber", ascending: false)]
        request.fetchLimit = 1
        
        do {
            let events = try shadowCoreDataManager.context.fetch(request)
            if let lastEvent = events.first {
                let sequence = UInt8(lastEvent.sequenceNumber)
                addDebugLog("📊 Last processed event sequence: \(sequence)")
                return sequence
            }
        } catch {
            addDebugLog("❌ Core Data error loading event sequence: \(error)")
        }
        
        addDebugLog("📊 No events found, starting event sequence from 0")
        return 0
    }
    
    private func saveLastKnownSequence(_ sequence: UInt8, for deviceId: UUID) {
        guard let shadowCoreDataManager = coreDataManager as? ShadowCoreDataManager else {
            addDebugLog("⚠️ Core Data manager not available, using fallback")
            lastSequenceNumber = sequence
            addDebugLog("💾 Saved sequence \(sequence) to memory (fallback)")
            return
        }
        
        let deviceIdentifier = deviceId.uuidString
        
        // Find or create the device record
        let request = NSFetchRequest<ShadowDevice>(entityName: "ShadowDevice")
        request.predicate = NSPredicate(format: "deviceIdentifier == %@", deviceIdentifier)
        request.fetchLimit = 1
        
        do {
            let devices = try shadowCoreDataManager.context.fetch(request)
            let device: ShadowDevice
            
            if let existingDevice = devices.first {
                device = existingDevice
                addDebugLog("📊 Found existing device record")
            } else {
                // Create new device record
                device = ShadowDevice(context: shadowCoreDataManager.context)
                device.deviceIdentifier = deviceIdentifier
                device.deviceName = "Shadow Device"
                device.lastKnownState = 0
                addDebugLog("📊 Created new device record")
            }
            
            // Update sequence and timestamp
            device.lastKnownSequence = Int16(sequence)
            device.lastConnectedDate = Date()
            
            shadowCoreDataManager.saveContext()
            addDebugLog("💾 Saved sequence \(sequence) for device \(deviceIdentifier)")
            
        } catch {
            addDebugLog("❌ Core Data error saving sequence: \(error)")
            // Fallback to memory storage
            lastSequenceNumber = sequence
        }
    }
    
    // MARK: - Power-Efficient Sync Protocol Functions
    
    private func performPostConnectionHandshake() {
        guard let peripheral = activePeripheral else {
            addDebugLog("❌ No active peripheral for handshake")
            return
        }
        
        addDebugLog("🤝 Starting post-connection handshake")
        currentSyncStatus = .synchronizing
        updateSystemStatus(.synchronizing)
        
        // Start by discovering services
        peripheral.discoverServices([stressServiceUUID])
    }
    
    private func finalizeSynchronizationAndDisconnect() {
        guard let peripheral = activePeripheral,
              let ackChar = ackCharacteristic else {
            addDebugLog("❌ Cannot finalize sync - missing peripheral or ACK characteristic")
            currentSyncStatus = .disconnected
            updateSystemStatus(.disconnected)
            return
        }
        
        // Send final acknowledgment with latest sequence
        let latestSequence = loadLastKnownEventSequence()
        let ackData = Data([latestSequence])
        peripheral.writeValue(ackData, for: ackChar, type: .withoutResponse)
        
        addDebugLog("✅ Sent final ACK with sequence \(latestSequence). Disconnecting immediately.")
        
        // Disconnect immediately without waiting for write confirmation
        centralManager.cancelPeripheralConnection(peripheral)
    }
    
    private func requestEventReplay(from sequence: UInt8) {
        guard let controlChar = controlPointCharacteristic,
              let peripheral = activePeripheral else {
            addDebugLog("❌ Cannot request replay - no control characteristic")
            finalizeSynchronizationAndDisconnect()
            return
        }
        
        let command = Data([ControlCommand.replayFromSequence.rawValue, sequence])
        peripheral.writeValue(command, for: controlChar, type: .withResponse)
        
        addDebugLog("🔄 Requested event replay from sequence \(sequence)")
        
        // Note: We'll receive events via eventCharacteristic notifications
        // and finalize sync after processing them
    }
    
    private func handleEventLogSequence(_ eventLogSequence: UInt8) {
        addDebugLog("📋 Handling event log sequence: \(eventLogSequence)")
        
        // Get last known event sequence from our database
        let lastKnownEventSequence = loadLastKnownEventSequence()
        
        if eventLogSequence > lastKnownEventSequence {
            // We need to catch up - request replay from our last known sequence
            addDebugLog("📥 Need to sync events from \(lastKnownEventSequence) to \(eventLogSequence)")
            requestEventReplay(from: lastKnownEventSequence + 1)
        } else {
            // We're up to date - can disconnect immediately
            addDebugLog("✅ Event log up to date (we have: \(lastKnownEventSequence), device has: \(eventLogSequence))")
            finalizeSynchronizationAndDisconnect()
        }
    }
    
    private func processReceivedEvent(_ data: Data) {
        // TODO: Process the actual stress event data
        // For now, just log that we received it
        addDebugLog("📨 Processing stress event (\(data.count) bytes)")
        
        // In a full implementation:
        // 1. Parse the stress_event_t structure
        // 2. Save to Core Data via ShadowCoreDataManager
        // 3. Update UI state
        
        // For now, just simulate processing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // Assume we've processed all events for now and finalize
            self.finalizeSynchronizationAndDisconnect()
        }
    }
    
    // MARK: - Discovery and Connection Functions
    
    func startScanning() {
        guard centralManager.state == .poweredOn else {
            addDebugLog("❌ Bluetooth not powered on")
            return
        }
        
        isScanning = true
        currentSyncStatus = .scanning
        updateSystemStatus(.scanning)
        connectionStatus = "Scanning for devices..."
        
        // Scan for devices advertising our stress service
        centralManager.scanForPeripherals(withServices: [stressServiceUUID], options: [
            CBCentralManagerScanOptionAllowDuplicatesKey: true
        ])
        
        addDebugLog("🔍 Started scanning for Shadow devices...")
    }
    
    func stopScanning() {
        centralManager.stopScan()
        isScanning = false
        currentSyncStatus = .disconnected
        updateSystemStatus(.disconnected)
        connectionStatus = "Disconnected"
        addDebugLog("⏹️ Stopped scanning")
    }
    
    func startContinuousScanning() {
        // Simply call startScanning() for continuous scanning
        startScanning()
    }
    
    func disconnect() {
        addDebugLog("🔌 Manually disconnecting from all devices")
        
        // Stop scanning first
        centralManager.stopScan()
        isScanning = false
        
        // Disconnect from all connected peripherals
        for peripheral in centralManager.retrieveConnectedPeripherals(withServices: [stressServiceUUID]) {
            addDebugLog("🔌 Disconnecting from \(peripheral.name ?? "Unknown")")
            centralManager.cancelPeripheralConnection(peripheral)
        }
        
        // Update status
        currentSyncStatus = .disconnected
        updateSystemStatus(.disconnected)
        connectionStatus = "Disconnected"
    }
    
    func connectToDevice(_ device: DiscoveredShadowDevice) {
        let peripheral = device.peripheral
        let sequence = device.advertisedSequence
        
        pendingSequenceNumber = sequence
        
        // Clean up any existing connection
        if let activePeripheral = activePeripheral {
            centralManager.cancelPeripheralConnection(activePeripheral)
        }
        
        // Set up new connection
        activePeripheral = peripheral
        connectedDevice = ShadowPeripheral(
            id: peripheral.identifier,
            name: peripheral.name ?? "Shadow Device",
            rssi: device.rssi,
            isConnected: false,
            lastSeen: Date(),
            advertisedState: device.advertisedState
        )
        serviceDiscoveryRetryCount = 0
        
        currentSyncStatus = .connecting
        updateSystemStatus(.connecting)
        connectionStatus = "Connecting..."
        
        addDebugLog("⚡ Starting manual connection for sequence \(sequence)")
        centralManager.connect(peripheral, options: nil)
    }
    
    // Power-efficient connection that immediately starts sync protocol
    private func initiateSequenceBasedConnection(to peripheral: CBPeripheral, expectedSequence sequence: UInt8) {
        pendingSequenceNumber = sequence
        
        // Clean up any existing connection
        if let activePeripheral = activePeripheral {
            centralManager.cancelPeripheralConnection(activePeripheral)
        }
        
        // Set up new connection
        activePeripheral = peripheral
        connectedDevice = ShadowPeripheral(
            id: peripheral.identifier,
            name: peripheral.name ?? "Shadow Device",
            rssi: -50, // We'll update this from advertisement
            isConnected: false,
            lastSeen: Date(),
            advertisedState: .disconnected // Default state, will be updated
        )
        serviceDiscoveryRetryCount = 0
        
        currentSyncStatus = .connecting
        updateSystemStatus(.connecting)
        
        addDebugLog("⚡ Starting Connect-Sync-Disconnect for sequence \(sequence)")
        centralManager.connect(peripheral, options: nil)
    }
    
    // MARK: - UI Helper Functions
    
    private func updateSystemStatus(_ status: SystemStatus) {
        DispatchQueue.main.async {
            self.currentSystemStatus = status
        }
    }
    
    private func addDebugLog(_ message: String) {
        let entry = DebugLogEntry(timestamp: Date(), message: message)
        
        DispatchQueue.main.async {
            self.debugLog.append(entry)
            // Keep only last 50 log entries
            if self.debugLog.count > 50 {
                self.debugLog.removeFirst(self.debugLog.count - 50)
            }
        }
    }
    
    func clearDebugLog() {
        DispatchQueue.main.async {
            self.debugLog.removeAll()
        }
    }
    
    private func updateDiscoveredDevices(peripheral: CBPeripheral, rssi: Int, advertisedState: SyncStatus, advertisedSequence: UInt8) {
        // Remove existing entry for this peripheral if it exists
        foundShadowDevices.removeAll { $0.peripheral.identifier == peripheral.identifier }
        
        // Add new entry
        let discoveredDevice = DiscoveredShadowDevice(
            peripheral: peripheral,
            rssi: rssi,
            advertisedState: advertisedState,
            advertisedSequence: advertisedSequence
        )
        foundShadowDevices.append(discoveredDevice)
        
        // Keep only the most recent 10 devices
        if foundShadowDevices.count > 10 {
            foundShadowDevices = Array(foundShadowDevices.suffix(10))
        }
        
        // Sort by most recent first
        foundShadowDevices.sort { $0.lastSeen > $1.lastSeen }
    }
    
    // MARK: - BLE Delegate Helper Functions
    
    private func handleFSMStateUpdate(_ data: Data) {
        // The FSM state notification contains the detailed event_log_sequence
        guard data.count >= 2 else { 
            addDebugLog("❌ FSM state data too small (need at least 2 bytes: state + sequence)")
            finalizeSynchronizationAndDisconnect()
            return 
        }
        
        // data[0] = FSM state, data[1] = event_log_sequence
        let fsmState = data[0]
        let eventLogSequence = data[1]
        addDebugLog("📋 FSM State: \(fsmState), Event Log Sequence: \(eventLogSequence)")
        
        // Handle the event log sequence according to protocol
        handleEventLogSequence(eventLogSequence)
    }
    
    private func handleEventDataUpdate(_ data: Data) {
        // This is called during event replay - process the received event
        guard data.count >= 16 else {  // Minimum size for stress_event_t
            addDebugLog("❌ Event data too small: \(data.count) bytes")
            return
        }
        
        totalEventsReceived += 1
        addDebugLog("📨 Received event #\(totalEventsReceived) during replay (\(data.count) bytes)")
        
        // Process the received event
        processReceivedEvent(data)
        
        // For now, finalize after receiving any event
        // TODO: Implement proper counting to know when all events are received
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.finalizeSynchronizationAndDisconnect()
        }
    }
}

// MARK: - CBCentralManagerDelegate

extension ShadowBLEManager: CBCentralManagerDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
            switch central.state {
            case .poweredOn:
                isBluetoothPoweredOn = true
                addDebugLog("✅ Bluetooth powered on")
                startScanning()
            case .poweredOff:
                isBluetoothPoweredOn = false
                addDebugLog("❌ Bluetooth powered off")
                currentSyncStatus = .disconnected
                updateSystemStatus(.disconnected)
            case .resetting:
                addDebugLog("🔄 Bluetooth resetting")
            case .unauthorized:
                addDebugLog("❌ Bluetooth unauthorized")
            case .unsupported:
                addDebugLog("❌ Bluetooth unsupported")
            case .unknown:
                addDebugLog("❓ Bluetooth state unknown")
            @unknown default:
                addDebugLog("❓ Unknown Bluetooth state")
            }
        }
    }
    
    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        Task { @MainActor in
            // Parse advertisement data to determine state and sequence
            var advertisedSequence: UInt8 = 0
            var advertisedState: SyncStatus = .scanning
            
            // Check if this advertisement contains sequence information
            if let serviceData = advertisementData[CBAdvertisementDataServiceDataKey] as? [CBUUID: Data],
               let stressData = serviceData[stressServiceUUID],
               stressData.count >= 1 {
                
                // Extract sequence from high 7 bits: (byte >> 1) & 0x7F
                let combinedByte = stressData[0]
                advertisedSequence = (combinedByte >> 1) & 0x7F
                advertisedState = .scanning // Could be parsed from advertisement if available
                let storedSequence = loadLastKnownSequence(for: peripheral.identifier)
                
                addDebugLog("📡 Found Shadow device: \(peripheral.name ?? "Unknown") (RSSI: \(RSSI))")
                addDebugLog("📊 Advertised sequence: \(advertisedSequence), Stored: \(storedSequence)")
                
                // Add or update device in discovered devices list
                updateDiscoveredDevices(peripheral: peripheral, rssi: RSSI.intValue, 
                                      advertisedState: advertisedState, advertisedSequence: advertisedSequence)
                
                // Check if we need to sync
                if advertisedSequence != storedSequence {
                    addDebugLog("🔄 Sequence change detected! Initiating sync...")
                    stopScanning()
                    initiateSequenceBasedConnection(to: peripheral, expectedSequence: advertisedSequence)
                } else {
                    addDebugLog("✅ No sync needed (sequences match)")
                }
            } else {
                addDebugLog("📡 Found device without sequence data: \(peripheral.name ?? "Unknown")")
                // Still add to discovered devices even without sequence data
                updateDiscoveredDevices(peripheral: peripheral, rssi: RSSI.intValue, 
                                      advertisedState: .scanning, advertisedSequence: 0)
            }
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            addDebugLog("✅ Connected to \(peripheral.name ?? "Unknown")")
            peripheral.delegate = self
            
            // Update connection state
            connectedDevice?.isConnected = true
            
            // Start the handshake process immediately
            performPostConnectionHandshake()
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Disconnected with error: \(error.localizedDescription)")
            } else {
                addDebugLog("✅ Disconnected successfully")
            }
            
            // Clean up state
            activePeripheral = nil
            connectedDevice?.isConnected = false
            connectedDevice = nil
            
            // Reset characteristics
            fsmCharacteristic = nil
            eventCharacteristic = nil
            controlPointCharacteristic = nil
            ackCharacteristic = nil
            
            // Update status and restart scanning
            currentSyncStatus = .disconnected
            updateSystemStatus(.disconnected)
            
            // Restart scanning after successful sync
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                if self.centralManager.state == .poweredOn {
                    self.startScanning()
                }
            }
        }
    }
    
    nonisolated func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        Task { @MainActor in
            addDebugLog("❌ Failed to connect: \(error?.localizedDescription ?? "Unknown error")")
            
            // Clean up and restart
            activePeripheral = nil
            connectedDevice = nil
            currentSyncStatus = .disconnected
            updateSystemStatus(.disconnected)
            
            startScanning()
        }
    }
}

// MARK: - CBPeripheralDelegate

extension ShadowBLEManager: CBPeripheralDelegate {
    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Service discovery error: \(error.localizedDescription)")
                finalizeSynchronizationAndDisconnect()
                return
            }
            
            guard let services = peripheral.services else {
                addDebugLog("❌ No services found")
                finalizeSynchronizationAndDisconnect()
                return
            }
            
            addDebugLog("🔍 Discovered \(services.count) services")
            
            // Find our stress service
            if let stressService = services.first(where: { $0.uuid == stressServiceUUID }) {
                addDebugLog("✅ Found stress service")
                peripheral.discoverCharacteristics([
                    fsmCharacteristicUUID,
                    eventCharacteristicUUID,
                    controlPointCharacteristicUUID,
                    ackCharacteristicUUID
                ], for: stressService)
            } else {
                addDebugLog("❌ Stress service not found")
                finalizeSynchronizationAndDisconnect()
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Characteristic discovery error: \(error.localizedDescription)")
                finalizeSynchronizationAndDisconnect()
                return
            }
            
            guard let characteristics = service.characteristics else {
                addDebugLog("❌ No characteristics found")
                finalizeSynchronizationAndDisconnect()
                return
            }
            
            addDebugLog("🔍 Discovered \(characteristics.count) characteristics")
            
            // Store characteristic references
            for characteristic in characteristics {
                switch characteristic.uuid {
                case fsmCharacteristicUUID:
                    fsmCharacteristic = characteristic
                    addDebugLog("✅ Found FSM characteristic")
                case eventCharacteristicUUID:
                    eventCharacteristic = characteristic
                    addDebugLog("✅ Found Event characteristic")
                case controlPointCharacteristicUUID:
                    controlPointCharacteristic = characteristic
                    addDebugLog("✅ Found Control Point characteristic")
                case ackCharacteristicUUID:
                    ackCharacteristic = characteristic
                    addDebugLog("✅ Found ACK characteristic")
                default:
                    break
                }
            }
            
            // Enable notifications on FSM and Event characteristics
            if let fsmChar = fsmCharacteristic {
                peripheral.setNotifyValue(true, for: fsmChar)
            }
            
            if let eventChar = eventCharacteristic {
                peripheral.setNotifyValue(true, for: eventChar)
            }
            
            // Start the sync protocol by sending handshake
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                self.sendHandshakeCommand()
            }
        }
    }
    
    private func sendHandshakeCommand() {
        guard let controlChar = controlPointCharacteristic,
              let peripheral = activePeripheral else {
            addDebugLog("❌ Cannot send handshake - no control characteristic")
            finalizeSynchronizationAndDisconnect()
            return
        }
        
        // Send "ACKNOWLEDGE_TRANSITION" command to trigger FSM notification
        // ESP32 expects 2 bytes: [opcode, sequence]
        guard let sequence = self.pendingSequenceNumber else {
            addDebugLog("❌ No pending sequence number for handshake")
            return
        }
        let handshakeCommand = Data([ControlCommand.acknowledgeTransition.rawValue, sequence])
        peripheral.writeValue(handshakeCommand, for: controlChar, type: .withResponse)
        
        addDebugLog("🤝 Sent handshake command")
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Characteristic update error: \(error.localizedDescription)")
                return
            }
            
            guard let data = characteristic.value else {
                addDebugLog("❌ No data received")
                return
            }
            
            switch characteristic.uuid {
            case fsmCharacteristicUUID:
                addDebugLog("📋 FSM state update received (\(data.count) bytes)")
                handleFSMStateUpdate(data)
            case eventCharacteristicUUID:
                addDebugLog("📨 Event data received (\(data.count) bytes)")
                handleEventDataUpdate(data)
            default:
                addDebugLog("📨 Update from unknown characteristic: \(characteristic.uuid)")
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Write error: \(error.localizedDescription)")
            } else {
                addDebugLog("✅ Write successful for \(characteristic.uuid)")
            }
        }
    }
    
    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateNotificationStateFor characteristic: CBCharacteristic, error: Error?) {
        Task { @MainActor in
            if let error = error {
                addDebugLog("❌ Notification setup error: \(error.localizedDescription)")
            } else {
                let state = characteristic.isNotifying ? "enabled" : "disabled"
                addDebugLog("✅ Notifications \(state) for \(characteristic.uuid)")
            }
        }
    }
}

// MARK: - ShadowPeripheral Model

struct ShadowPeripheral: Identifiable {
    let id: UUID
    let name: String
    let rssi: Int
    var isConnected: Bool
    let lastSeen: Date
    let advertisedState: SyncStatus
}

// MARK: - Debug Helpers

private extension DateFormatter {
    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss.SSS"
        return formatter
    }()
}

// MARK: - CoreDataManager Protocol

protocol CoreDataManager {
    func saveContext()
}
