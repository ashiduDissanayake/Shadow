//
//  QRScannerView.swift
//  Shadow
//
//  Created on 19/10/2025.
//  QR Code Scanner for Shadow Device Pairing
//  REQUIRES: Real Mac with camera (not simulator)
//

import SwiftUI
import AVFoundation
import Vision

struct QRScannerView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var scanner = QRScanner()
    var onDeviceScanned: ((String) -> Void)?
    
    @State private var scannedCode: String?
    @State private var showError = false
    @State private var errorMessage = ""
    @State private var cameraStatus = "Initializing..."
    
    var body: some View {
        ZStack {
            // Camera preview
            QRScannerCameraView(scanner: scanner)
                .edgesIgnoringSafeArea(.all)
            
            // Top overlay with title and close button
            VStack {
                HStack {
                    Text("Scan Device QR Code")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(10)
                    
                    Spacer()
                    
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .clipShape(Circle())
                    }
                }
                .padding()
                
                Spacer()
                
                // Scanning indicator
                if scanner.captureSession?.isRunning == true {
                    ScanningIndicator()
                        .padding(.bottom, 20)
                }
                
                // Instructions
                Text("Align QR code within frame")
                    .font(.subheadline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding(.bottom, 20)
                
                // Debug status
                Text(cameraStatus)
                    .font(.caption)
                    .foregroundColor(.yellow)
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                    .padding(.bottom, 20)
                
                // Scanning frame
                Rectangle()
                    .stroke(Color.green, lineWidth: 3)
                    .frame(width: 250, height: 250)
                    .overlay(
                        VStack {
                            HStack {
                                Rectangle().fill(Color.green).frame(width: 20, height: 3)
                                Spacer()
                                Rectangle().fill(Color.green).frame(width: 20, height: 3)
                            }
                            Spacer()
                            HStack {
                                Rectangle().fill(Color.green).frame(width: 20, height: 3)
                                Spacer()
                                Rectangle().fill(Color.green).frame(width: 20, height: 3)
                            }
                        }
                        .frame(width: 250, height: 250)
                    )
                    .padding(.bottom, 80)
            }
        }
        .alert("Invalid QR Code", isPresented: $showError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .onAppear {
            cameraStatus = "Starting camera..."
            print("🎥 QRScannerView appeared - starting camera")
            scanner.startScanning(
                onStatusUpdate: { status in
                    cameraStatus = status
                    print("📹 Camera status: \(status)")
                },
                onCodeScanned: { code in
                    print("📱 QR Code detected: \(code)")
                    handleScannedCode(code)
                }
            )
        }
        .onDisappear {
            print("🎥 QRScannerView disappeared - stopping camera")
            scanner.stopScanning()
        }
    }
    
    // MARK: - QR Code Handler
    private func handleScannedCode(_ code: String) {
        // Expected format: "Shadow-XXXX" (device name only)
        guard code.hasPrefix("Shadow-") else {
            errorMessage = "Invalid Shadow device QR code. Expected format: Shadow-XXXX"
            showError = true
            return
        }
        
        scannedCode = code
        
        // Save paired device
        UserDefaults.standard.set(code, forKey: "PairedShadowDevice")
        
        // Call callback if provided
        onDeviceScanned?(code)
        
        // Dismiss scanner
        dismiss()
    }
}

// MARK: - Camera Preview View

struct QRScannerCameraView: NSViewRepresentable {
    @ObservedObject var scanner: QRScanner
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.layer = CALayer()
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.black.cgColor
        
        print("🖼️ QRScannerCameraView: makeNSView called")
        
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        // Remove old preview layer if exists
        nsView.layer?.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        if let previewLayer = scanner.previewLayer {
            print("🖼️ QRScannerCameraView: Adding preview layer to view")
            previewLayer.frame = nsView.bounds
            nsView.layer?.addSublayer(previewLayer)
        } else {
            print("🖼️ QRScannerCameraView: No preview layer available yet")
        }
    }
}

// MARK: - QR Scanner Manager

class QRScanner: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var captureSession: AVCaptureSession?
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    private var onCodeScanned: ((String) -> Void)?
    private var onStatusUpdate: ((String) -> Void)?
    private var isProcessing = false
    private let visionQueue = DispatchQueue(label: "com.shadow.qr.vision")
    
    func startScanning(onStatusUpdate: @escaping (String) -> Void, onCodeScanned: @escaping (String) -> Void) {
        self.onStatusUpdate = onStatusUpdate
        self.onCodeScanned = onCodeScanned
        
        print("🎬 Starting QR scanner setup...")
        onStatusUpdate("Requesting camera access...")
        
        // Request camera permission
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                if granted {
                    print("✅ Camera permission granted")
                    onStatusUpdate("Setting up camera...")
                    self.setupCamera()
                } else {
                    print("❌ Camera permission denied")
                    onStatusUpdate("⚠️ Camera permission denied - Please enable in System Preferences")
                }
            }
        }
    }
    
    private func setupCamera() {
        onStatusUpdate?("Initializing camera...")
        
        captureSession = AVCaptureSession()
        
        guard let captureSession = captureSession else {
            print("❌ Failed to create capture session")
            onStatusUpdate?("❌ Failed to create capture session")
            return
        }
        
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else {
            print("❌ No camera device found")
            onStatusUpdate?("❌ No camera found")
            return
        }
        
        print("📹 Found camera device: \(videoCaptureDevice.localizedName)")
        
        guard let videoInput = try? AVCaptureDeviceInput(device: videoCaptureDevice) else {
            print("❌ Failed to create video input")
            onStatusUpdate?("❌ Failed to access camera")
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
            print("✅ Added video input")
        } else {
            print("❌ Cannot add video input")
            onStatusUpdate?("❌ Cannot add video input")
            return
        }
        
        // Use video data output instead of metadata (for Vision framework)
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            print("✅ Added video output for Vision framework QR detection")
        } else {
            print("❌ Cannot add video output")
            onStatusUpdate?("❌ Cannot add video output")
            return
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        print("✅ Created preview layer")
        
        onStatusUpdate?("Starting camera feed...")
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            captureSession.startRunning()
            print("✅ Camera session started")
            
            DispatchQueue.main.async {
                self?.onStatusUpdate?("✅ Camera ready - Scan QR code")
            }
        }
    }
    
    func stopScanning() {
        captureSession?.stopRunning()
        captureSession = nil
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    func captureOutput(_ output: AVCaptureOutput,
                      didOutput sampleBuffer: CMSampleBuffer,
                      from connection: AVCaptureConnection) {
        
        // Avoid processing multiple frames at once
        guard !isProcessing else { return }
        isProcessing = true
        
        defer { isProcessing = false }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Create Vision request for QR code detection
        let request = VNDetectBarcodesRequest { [weak self] request, error in
            guard let self = self else { return }
            
            if let error = error {
                print("📸 Vision error: \(error.localizedDescription)")
                return
            }
            
            guard let results = request.results as? [VNBarcodeObservation] else {
                return
            }
            
            // Process QR codes
            for observation in results {
                if observation.symbology == .qr,
                   let payloadString = observation.payloadStringValue {
                    
                    print("📸 QR Code detected via Vision: \(payloadString)")
                    
                    // Stop scanning and notify
                    DispatchQueue.main.async {
                        self.stopScanning()
                        
                        // Haptic feedback
                        NSSound.beep()
                        
                        print("📸 Calling onCodeScanned callback")
                        self.onCodeScanned?(payloadString)
                    }
                    
                    return
                }
            }
        }
        
        // Perform the request
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("📸 Failed to perform Vision request: \(error.localizedDescription)")
        }
    }
}

// MARK: - Scanning Indicator

struct ScanningIndicator: View {
    @State private var isPulsing = false
    
    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color.red)
                .frame(width: 8, height: 8)
                .scaleEffect(isPulsing ? 1.5 : 1.0)
                .opacity(isPulsing ? 0.3 : 1.0)
                .animation(
                    Animation.easeInOut(duration: 1.0)
                        .repeatForever(autoreverses: true),
                    value: isPulsing
                )
            
            Text("SCANNING")
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(.red)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.black.opacity(0.7))
        .cornerRadius(20)
        .onAppear {
            isPulsing = true
        }
    }
}

// MARK: - Preview

struct QRScannerView_Previews: PreviewProvider {
    static var previews: some View {
        QRScannerView(onDeviceScanned: { deviceName in
            print("Preview: Scanned \(deviceName)")
        })
    }
}
