//
//  ESPLink.swift
//  Shadow
//
//  Created by Ashidu Dissanayake on 2025-08-18.
//


import Foundation
import Network
import UserNotifications
import os.log

final class ESPLink: ObservableObject {
    private let logger = Logger(subsystem: "ESPLink", category: "net")
    private let queue = DispatchQueue(label: "esp.link.queue", qos: .utility)
    private var listener: NWListener?
    private var connections = [UUID: Connection]()
    @Published var connectedCount: Int = 0

    struct Connection {
        let id: UUID
        let nw: NWConnection
        var buffer = Data()
    }

    init(port: UInt16 = 5001) {
        requestNotificationPermission()
        start(port: port)
    }

    deinit { stop() }

    func start(port: UInt16) {
        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            let listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            listener.service = NWListener.Service(name: Host.current().localizedName ?? "ESP Link",
                                                  type: "_espbridge._tcp")
            listener.newConnectionHandler = { [weak self] conn in
                self?.accept(conn)
            }
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready: self.log("Listener ready on :\(port)")
                case .failed(let err): self.log("Listener failed: \(err.localizedDescription)")
                default: break
                }
            }
            listener.start(queue: queue)
            self.listener = listener
        } catch {
            log("Failed to start listener: \(error.localizedDescription)")
        }
    }

    func stop() {
        listener?.cancel()
        connections.values.forEach { $0.nw.cancel() }
        connections.removeAll()
        connectedCount = 0
    }

    private func accept(_ nw: NWConnection) {
        let id = UUID()
        let c = Connection(id: id, nw: nw)
        connections[id] = c
        connectedCount = connections.count

        nw.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.log("Connection ready: \(id)")
                self?.receiveLoop(id: id)
            case .failed(let err):
                self?.log("Connection closed: \(id) err=\(err.localizedDescription)")
                self?.connections.removeValue(forKey: id)
                self?.connectedCount = self?.connections.count ?? 0

            case .cancelled:
                self?.log("Connection cancelled: \(id)")
                self?.connections.removeValue(forKey: id)
                self?.connectedCount = self?.connections.count ?? 0
            default: break
            }
        }
        nw.start(queue: queue)
    }

    private func receiveLoop(id: UUID) {
        guard let conn = connections[id] else { return }
        conn.nw.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }
            if let data = data, !data.isEmpty {
                self.appendAndProcess(id: id, data: data)
            }
            if isComplete || error != nil {
                self.log("Receive ended: \(id)")
                self.connections[id]?.nw.cancel()
                self.connections.removeValue(forKey: id)
                self.connectedCount = self.connections.count
                return
            }
            self.receiveLoop(id: id)
        }
    }

    private func appendAndProcess(id: UUID, data: Data) {
        guard var conn = connections[id] else { return }
        conn.buffer.append(data)
        while let range = conn.buffer.firstRange(of: Data([0x0A])) { // newline
            let line = conn.buffer.subdata(in: conn.buffer.startIndex..<range.lowerBound)
            conn.buffer.removeSubrange(conn.buffer.startIndex..<range.upperBound)
            handleLine(id: id, line: line)
        }
        connections[id] = conn
    }

    private func handleLine(id: UUID, line: Data) {
        guard let text = String(data: line, encoding: .utf8) else { return }
        log("RX: \(text)")
        if let json = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any] {
            if let type = json["type"] as? String, type == "telemetry" {
                postNotification(title: json["name"] as? String ?? "ESP32",
                                 body: "Telemetry: \(json)")
            }
        }
    }

    // MARK: - Sending
    func send(json: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: json),
              let msg = String(data: data, encoding: .utf8)?
                .appending("\n").data(using: .utf8) else { return }
        connections.values.forEach { $0.nw.send(content: msg, completion: .contentProcessed { _ in }) }
        log("TX: \(json)")
    }

    // MARK: - Notifications
    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { _, _ in }
    }

    private func postNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        let req = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(req, withCompletionHandler: nil)
    }

    private func log(_ s: String) { logger.log("\(s, privacy: .public)") }
}
