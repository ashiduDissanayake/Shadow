//
//  DeviceSettingsView.swift
//  Shadow
//
//  Created on 19/10/2025.
//  Device pairing and management view
//

import SwiftUI

struct DeviceSettingsView: View {
    @ObservedObject var syncViewModel: SyncDashboardViewModel
    
    @State private var showQRScanner = false
    @State private var showUnpairConfirmation = false
    @State private var pairedDevice: String?
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Shadow Device")
                .font(.title)
                .fontWeight(.bold)
            
            Divider()
            
            // Paired Device Section
            if let device = syncViewModel.manager.pairedDeviceName {
                // Device is paired - show controls
                pairedDeviceView(device: device)
            } else {
                // No device paired - show pairing UI
                unpairedDeviceView
            }
            
            Spacer()
        }
        .padding()
        .sheet(isPresented: $showQRScanner) {
            QRScannerView(onDeviceScanned: { deviceName in
                print("✅ Device paired: \(deviceName)")
                pairedDevice = deviceName
                showQRScanner = false
                syncViewModel.start()
            })
        }
        .alert("Forget Device?", isPresented: $showUnpairConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Forget", role: .destructive) {
                syncViewModel.stop()
                syncViewModel.manager.unpairDevice()
                pairedDevice = nil
            }
        } message: {
            Text("This will remove \(syncViewModel.manager.pairedDeviceName ?? "the device"). You'll need to scan the QR code again to reconnect.")
        }
        .onAppear {
            pairedDevice = syncViewModel.manager.pairedDeviceName
        }
    }
    
    // MARK: - Paired Device View
    private func pairedDeviceView(device: String) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 20) {
                // Device Info Header
                HStack {
                    Image(systemName: "antenna.radiowaves.left.and.right.circle.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.blue)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        Text(device)
                            .font(.title3)
                            .fontWeight(.bold)
                        
                        HStack(spacing: 6) {
                            Circle()
                                .fill(syncViewModel.isActive ? Color.green : Color.gray)
                                .frame(width: 10, height: 10)
                            Text(syncViewModel.isActive ? "Syncing" : "Idle")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                }
                
                Divider()
                
                // Sync Statistics
                VStack(spacing: 12) {
                    statisticRow(label: "Last Sync", value: syncViewModel.lastSync)
                    statisticRow(label: "Events Received", value: "\(syncViewModel.eventsReceived)")
                    statisticRow(
                        label: "Current State",
                        value: syncViewModel.currentStateLabel,
                        valueColor: syncViewModel.currentStateLabel == "STRESS" ? .red : .green
                    )
                }
                
                Divider()
                
                // Action Buttons
                HStack(spacing: 12) {
                    if syncViewModel.isActive {
                        Button(action: { syncViewModel.stop() }) {
                            Label("Stop Sync", systemImage: "pause.circle.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .tint(.orange)
                    } else {
                        Button(action: { syncViewModel.start() }) {
                            Label("Start Sync", systemImage: "play.circle.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    
                    Button(action: { showUnpairConfirmation = true }) {
                        Label("Forget", systemImage: "trash")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }
            .padding()
        }
    }
    
    // MARK: - Unpaired Device View
    private var unpairedDeviceView: some View {
        GroupBox {
            VStack(spacing: 24) {
                Image(systemName: "qrcode.viewfinder")
                    .font(.system(size: 64))
                    .foregroundColor(.blue.opacity(0.6))
                
                VStack(spacing: 8) {
                    Text("No Device Paired")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    Text("Scan the QR code displayed on your Shadow device to pair and start monitoring.")
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                Button(action: { showQRScanner = true }) {
                    Label("Scan QR Code", systemImage: "qrcode")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding(24)
        }
    }
    
    // MARK: - Helper Views
    private func statisticRow(label: String, value: String, valueColor: Color = .primary) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(valueColor)
        }
    }
}

// Preview
struct DeviceSettingsView_Previews: PreviewProvider {
    static var previews: some View {
        DeviceSettingsView(syncViewModel: SyncDashboardViewModel())
    }
}
