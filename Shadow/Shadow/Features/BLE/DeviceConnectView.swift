import SwiftUI

struct DeviceConnectView: View {
    @ObservedObject var syncViewModel: SyncDashboardViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                Text("Shadow Automatic Monitor")
                    .font(.title2.bold())
                    .foregroundColor(.white)

                // Sync status card
                syncStatusCard
                
                // System data card  
                systemDataCard
                
                Spacer()
            }
            .padding()
            .background(shadowBackground)
            .navigationTitle("Shadow Sync Monitor")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("Done") {
                        dismiss()
                    }
                    .foregroundColor(.white)
                }
            }
        }
        .onAppear {
            // Auto-start syncing when view appears
            if !syncViewModel.isActive {
                syncViewModel.start()
            }
        }
    }
    
    private var syncStatusCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Circle()
                    .fill(syncStatusColor)
                    .frame(width: 12, height: 12)
                
                Text("Sync Status")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
                
                Text(syncViewModel.stateText)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            // Sync info
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Last Sync:")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                    
                    Spacer()
                    
                    Text(syncViewModel.lastSync)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
                
                HStack {
                    Text("Sequence:")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                    
                    Spacer()
                    
                    Text(syncViewModel.sequenceStatus)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
            }
            
            Text(syncDescription)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var systemDataCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(.green)
                    .frame(width: 12, height: 12)
                
                Text("System Data")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
            }
            
            VStack(alignment: .leading, spacing: 8) {
                statusRow("Status", syncViewModel.stateText)
                statusRow("Events Received", "\(syncViewModel.eventsReceived)")
                statusRow("Sequence", syncViewModel.sequenceStatus)
                statusRow("Last Sync", syncViewModel.lastSync)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(.ultraThinMaterial)
        )
    }
    
    private var syncDescription: String {
        if syncViewModel.isActive {
            return "Currently synchronizing with Shadow device. Real-time stress detection active."
        } else if syncViewModel.stateText == "Up to Date" {
            return "System is synchronized and up to date. Ready for stress detection."
        } else {
            return "Power-efficient monitoring mode. Tap refresh to check for updates."
        }
    }
    
    private var syncStatusColor: Color {
        if syncViewModel.isActive {
            return .orange
        } else if syncViewModel.stateText == "Up to Date" {
            return .green
        } else {
            return .gray
        }
    }
    
    private var shadowBackground: some View {
        LinearGradient(
            gradient: Gradient(colors: [
                Color(red: 0.05, green: 0.08, blue: 0.15),
                Color(red: 0.1, green: 0.15, blue: 0.25)
            ]),
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
    
    private func statusRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
            
            Spacer()
            
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
    }
}
