import SwiftUI
import Charts
import CoreData

struct StressPoint: Identifiable {
    let id = UUID()
    let date: Date
    let state: Int // 0 or 1
    let seq: Int
}

struct StressStateGraphView: View {
    let events: [StressPoint]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with event count
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Stress Timeline")
                            .font(.title2)
                            .fontWeight(.bold)
                        Text("\(events.count) event\(events.count == 1 ? "" : "s") recorded")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Close") { dismiss() }
                        .buttonStyle(.bordered)
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                if events.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "chart.line.uptrend.xyaxis")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No stress events yet")
                            .font(.headline)
                        Text("Events will appear here once the device reports state transitions.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    .frame(maxHeight: .infinity)
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        // Legend
                        HStack(spacing: 16) {
                            HStack(spacing: 4) {
                                Circle().fill(Color.red).frame(width: 10, height: 10)
                                Text("Stressed")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            HStack(spacing: 4) {
                                Circle().fill(Color.green).frame(width: 10, height: 10)
                                Text("Calm")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Text("Last 24 hours")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                        .padding(.top, 12)
                        
                        chartView
                            .frame(height: 400)
                            .padding()
                    }
                }

                Spacer()
            }
        }
    }
    
    // Extracted chart view to fix type-checking timeout
    private var chartView: some View {
        StressStateGraphView.chartView(for: events)
    }
    
    // Static chart view for embedding in other views
    static func chartView(for events: [StressPoint]) -> some View {
        Chart {
            ForEach(events) { e in
                // Larger point marks
                PointMark(x: .value("Time", e.date), y: .value("State", e.state))
                    .symbolSize(100)
                    .foregroundStyle(e.state == 1 ? Color.red : Color.green)
                
                // Line connecting points
                LineMark(x: .value("Time", e.date), y: .value("State", e.state))
                    .foregroundStyle(e.state == 1 ? Color.red.opacity(0.3) : Color.green.opacity(0.3))
                    .lineStyle(StrokeStyle(lineWidth: 2))
            }
        }
                .chartXAxis {
            AxisMarks(values: .stride(by: .hour)) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel(format: .dateTime.hour().minute())
            }
        }
        .chartYAxis {
            AxisMarks(values: [0, 1]) { value in
                AxisGridLine()
                AxisTick()
                AxisValueLabel {
                    if let intValue = value.as(Int.self) {
                        Text(intValue == 1 ? "Stressed" : "Calm")
                            .font(.caption)
                    }
                }
            }
        }
        .chartYScale(domain: -0.2...1.2)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.2))
        )
    }
}

// Helper to convert CoreData StressEvent -> StressPoint
extension StressStateGraphView {
    static func fromCoreData(_ events: [StressEvent]) -> [StressPoint] {
        events.map { e in
            let date = e.timestamp ?? Date()
            let state = Int(e.stressState)  // CoreData attribute is 'stressState'
            let seq = Int(e.sequenceNumber)  // CoreData attribute is 'sequenceNumber'
            return StressPoint(date: date, state: state, seq: seq)
        }
        .sorted { $0.date < $1.date }
    }
}

struct StressStateGraphView_Previews: PreviewProvider {
    static var previews: some View {
        StressStateGraphView(events: [
            StressPoint(date: Date().addingTimeInterval(-3600*3), state: 0, seq: 1),
            StressPoint(date: Date().addingTimeInterval(-3600*2), state: 1, seq: 2),
            StressPoint(date: Date().addingTimeInterval(-3600*1), state: 0, seq: 3)
        ])
    }
}
