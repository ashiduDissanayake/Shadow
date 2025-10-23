import SwiftUI
import CoreData

struct RecentStressEventRow: View {
    let event: StressEvent
    
    var body: some View {
        HStack {
            // State indicator
            RoundedRectangle(cornerRadius: 3)
                .fill(stateColor)
                .frame(width: 6, height: 20)
            
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text("Seq: \(event.sequenceNumber)")
                        .font(.caption2)
                        .foregroundColor(.shadowTextSecondary)
                    
                    Spacer()
                    
                    Text(stateText)
                        .font(.caption2)
                        .fontWeight(.medium)
                        .foregroundColor(stateColor)
                }
                
                Text(timeAgo(event.timestamp ?? Date()))
                    .font(.caption2)
                    .foregroundColor(.shadowTextTertiary)
            }
            
            Spacer()
            
            if event.confidenceScore > 0 {
                Text("\(Int(event.confidenceScore * 100))%")
                    .font(.caption2)
                    .foregroundColor(.shadowTextPrimary)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.shadowBackgroundSecondary)
                    )
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.shadowBackgroundSecondary.opacity(0.8))
        )
    }
    
    private var stateColor: Color {
        switch event.stressState {
        case 0: return .shadowSuccess
        case 1: return .shadowStressHigh
        default: return .shadowTextTertiary
        }
    }
    
    private var stateText: String {
        switch event.stressState {
        case 0: return "Calm"
        case 1: return "Stress"
        default: return "Unknown"
        }
    }
    
    private func timeAgo(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}
