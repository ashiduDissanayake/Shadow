# Adding Pairing UI to Shadow macOS App - Quick Guide

**File to Edit**: `/Shadow/Shadow/Features/Dashboard/ShadowDashboardView.swift`

---

## 🎯 **Option 1: Add Pairing Button to Shadow Status Section** (Recommended)

Add a pairing button below the existing status rows in the `shadowStatusSection`.

### **Step 1: Add State Variables**

Add these after the existing `@State` variables (around line 13):

```swift
@State private var showingDebugLog = false
@State private var showingCoreDataDebug = false
@State private var recentEvents: [StressEvent] = []

// ADD THESE:
@State private var showingPairingAlert = false
@State private var pairingErrorMessage: String?
```

### **Step 2: Update Status Section**

Find the `shadowStatusSection` variable (around line 86) and add the pairing button after the status rows:

```swift
private var shadowStatusSection: some View {
    VStack(alignment: .leading, spacing: 16) {
        HStack {
            Image(systemName: "brain.head.profile")
                .font(.title2)
                .foregroundColor(.blue)
            Text("Shadow Monitoring")
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(.white)
            Spacer()
            statusIndicator
        }
        
        VStack(spacing: 12) {
            statusRow("System Status", syncViewModel.stateText, systemColor: systemStatusColor)
            statusRow("Last Sync", syncViewModel.lastSync)
            statusRow("Sequence", syncViewModel.sequenceStatus)
            
            Divider()
                .background(Color.white.opacity(0.3))
            
            // ADD PAIRING SECTION HERE:
            pairingSection
        }
    }
    .padding()
    .background(
        RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
    )
}
```

### **Step 3: Add Pairing Section View**

Add this new computed property after `shadowStatusSection`:

```swift
// MARK: Pairing Section
private var pairingSection: some View {
    VStack(alignment: .leading, spacing: 12) {
        HStack {
            Image(systemName: "lock.shield")
                .foregroundColor(syncViewModel.manager.isPaired ? .green : .orange)
            
            VStack(alignment: .leading, spacing: 4) {
                Text("Device Pairing")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                if let deviceInfo = syncViewModel.manager.deviceInfo {
                    Text("\(deviceInfo.deviceName) - \(deviceInfo.firmwareVersion)")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                } else {
                    Text("Not paired")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            
            Spacer()
            
            // Pairing state indicator
            HStack(spacing: 4) {
                Text(syncViewModel.manager.pairingState.emoji)
                Text(syncViewModel.manager.pairingState.description)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
            }
        }
        
        // Pairing button
        if !syncViewModel.manager.isPaired {
            Button(action: {
                Task {
                    do {
                        try await syncViewModel.manager.performPairing()
                        showingPairingAlert = true
                        pairingErrorMessage = nil
                    } catch {
                        pairingErrorMessage = error.localizedDescription
                        showingPairingAlert = true
                    }
                }
            }) {
                HStack {
                    Image(systemName: "key.fill")
                    Text("Pair Device")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 8)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .buttonStyle(.plain)
            .disabled(syncViewModel.manager.pairingState == .pending)
        } else {
            // Show paired status
            HStack {
                Image(systemName: "checkmark.shield.fill")
                    .foregroundColor(.green)
                Text("Device Paired")
                    .fontWeight(.semibold)
                    .foregroundColor(.green)
                Spacer()
                Button("Unpair") {
                    // TODO: Implement unpair functionality if needed
                    syncViewModel.manager.isPaired = false
                    syncViewModel.manager.deviceInfo = nil
                }
                .font(.caption)
                .foregroundColor(.red.opacity(0.8))
            }
            .padding(.vertical, 8)
        }
    }
}
```

### **Step 4: Add Alert Modifier**

Add this to the main `body` after the existing `.sheet` modifiers (around line 50):

```swift
.sheet(isPresented: $showingCoreDataDebug) {
    CoreDataDebugView()
}
// ADD THIS:
.alert("Pairing", isPresented: $showingPairingAlert) {
    Button("OK") { }
} message: {
    if let errorMessage = pairingErrorMessage {
        Text("Pairing failed: \(errorMessage)")
    } else {
        Text("Device paired successfully! ✅")
    }
}
.onAppear {
    syncViewModel.start()
    recentEvents = syncViewModel.getRecentEvents()
}
```

### **Step 5: Expose Manager in ViewModel**

Edit `/Shadow/Shadow/Features/BLE/SyncDashboardViewModel.swift` to expose the manager:

```swift
@MainActor
final class SyncDashboardViewModel: ObservableObject {
    @Published var stateText: String = "Idle"
    @Published var lastSync: String = "-"
    @Published var sequenceStatus: String = "-"
    @Published var log: [String] = []
    @Published var eventsReceived: Int = 0
    @Published var isActive: Bool = false
    @Published var currentStateLabel: String = "CALM"
    
    // MAKE THIS PUBLIC:
    let manager: LightShadowBLEManager  // Change from 'private' to public
    
    // ... rest of the code
}
```

---

## 🎨 **Option 2: Simpler Floating Pairing Button** (Alternative)

If you want a simpler approach, just add a floating button in the header:

### **In headerSection, add button next to profile:**

```swift
private var headerSection: some View {
    VStack(alignment: .leading, spacing: 8) {
        HStack {
            Text("Welcome back, \(profile.name ?? "User")!")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Spacer()
            
            // ADD PAIRING BUTTON:
            if !syncViewModel.manager.isPaired {
                Button(action: {
                    Task {
                        try? await syncViewModel.manager.performPairing()
                    }
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "key.fill")
                        Text("Pair")
                    }
                    .font(.caption)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.shield.fill")
                        .foregroundColor(.green)
                    Text("Paired")
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }
            
            Button(action: onShowProfile) {
                Image(systemName: "person.circle")
                    .font(.title2)
                    .foregroundColor(.white.opacity(0.8))
            }
        }
        Text("Shadow stress monitoring dashboard")
            .font(.subheadline)
            .foregroundColor(.white.opacity(0.7))
    }
    .padding()
    .background(
        RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
    )
}
```

---

## 🧪 **Testing Steps**

1. **Build and Run** the Shadow app in Xcode
2. **Make sure** Shadow-9026 ESP32 is running (`idf.py monitor`)
3. **Login** to the app with your credentials
4. **Navigate** to the Shadow Dashboard
5. **Click** "Pair Device" button
6. **Watch** the pairing flow:
   - Button should show loading state
   - Pairing state should change: Idle → Connected → Pending → Paired
   - Alert should show "Device paired successfully!"
7. **Check** ESP32 monitor logs for pairing confirmation
8. **Verify** device info appears (Shadow-9026, v1.0.0, ESP32-S3)
9. **Restart** app and verify pairing persists

---

## 📊 **Expected UI Flow**

### **Before Pairing:**
```
┌────────────────────────────────────┐
│ Device Pairing                     │
│ Not paired                         │
│                                    │
│ 📡 Advertising                     │
│                                    │
│ ┌──────────────────────────────┐  │
│ │ 🔑  Pair Device               │  │
│ └──────────────────────────────┘  │
└────────────────────────────────────┘
```

### **During Pairing:**
```
┌────────────────────────────────────┐
│ Device Pairing                     │
│ Shadow-9026 - v1.0.0               │
│                                    │
│ ⏳ Pending                         │
│                                    │
│ (Button disabled)                  │
└────────────────────────────────────┘
```

### **After Pairing:**
```
┌────────────────────────────────────┐
│ Device Pairing                     │
│ Shadow-9026 - v1.0.0               │
│                                    │
│ ✅ Paired                          │
│                                    │
│ ✅ Device Paired      [Unpair]     │
└────────────────────────────────────┘
```

---

## 🚀 **Ready to Test!**

Choose **Option 1** for a comprehensive pairing section with full status display, or **Option 2** for a minimal floating button approach.

Both options use the same underlying `performPairing()` method, so the functionality is identical! 🎉
