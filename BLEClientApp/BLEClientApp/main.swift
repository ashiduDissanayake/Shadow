import Foundation

let bleClient = BLEClient()

bleClient.onConnected = {
    print("BLE Client: Connected to ESP32!")
    // Example: Send a control command after connection
    bleClient.writeControlCommand(command: "start_data")
    // Example: Send some data
    bleClient.writeData(dataString: "Hello from Mac!")
}

bleClient.onDisconnected = {
    print("BLE Client: Disconnected from ESP32.")
}

bleClient.onDataReceived = { data in
    print("BLE Client: Received data: \(data)")
}

bleClient.onStatusReceived = { status in
    print("BLE Client: Received status: \(status)")
}

print("Starting BLE client...")
bleClient.startScanning()

// Keep the program running to allow BLE operations
RunLoop.main.run()


