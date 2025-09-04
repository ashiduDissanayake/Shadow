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
                        .foregroundColor(.white.opacity(0.9))
                    
                    Spacer()
                    
                    Text(stateText)
                        .font(.caption2)
                        .fontWeight(.medium)
                        .foregroundColor(stateColor)
                }
                
                Text(timeAgo(event.timestamp ?? Date()))
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.6))
            }
            
            Spacer()
            
            if event.confidenceScore > 0 {
                Text("\(Int(event.confidenceScore * 100))%")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.7))
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.black.opacity(0.3))
                    )
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(.black.opacity(0.2))
        )
    }
    
    private var stateColor: Color {
        switch event.stressState {
        case 0: return .green
        case 1: return .red
        default: return .gray
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
