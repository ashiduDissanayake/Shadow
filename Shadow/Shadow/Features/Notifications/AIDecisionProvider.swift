import Foundation

final class AIDecisionProvider {
    static let shared = AIDecisionProvider()
    private let apiKey = "AIzaSyD6pKke1P9h_bItwm_cT7_mcw4cDCNw5lU"
    private let endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    
    private init() {}

    /// Generate contextual message using Gemini 2.0 Flash
    func message(for event: StressEvent) async -> String {
        let state = event.stressState
        let timestamp = event.timestamp ?? Date()
        let timeOfDay = Calendar.current.component(.hour, from: Date())
        
        // Create a detailed prompt with context
        let prompt: String
        if state == 1 {
            // STRESS DETECTED - Focus on actionable micro-interventions
            prompt = """
            You're a mindfulness coach. A user's stress monitor detected elevated stress.
            
            Context:
            - Time of day: \(timeOfDay):00 (\(timeOfDayContext(hour: timeOfDay)))
            - Detection time: \(timestamp.formatted())
            
            Generate a SHORT, personal notification (MAX 35 words) that:
            1. NO generic statements like "stress detected" or "we noticed"
            2. Offer ONE specific micro-action they can do RIGHT NOW (breathing, quick walk, water break, stretch)
            3. Warm tone, second person ("you"), like texting a friend
            4. NO markdown, NO emojis, NO asterisks
            
            Good examples:
            - "Quick reset: Close your eyes. Breathe in for 4, hold for 4, out for 6. Three times."
            - "Try this: Stand up, roll your shoulders back three times, take a deep breath."
            - "Water break time. While you drink, do three slow, deep breaths."
            
            Bad examples:
            - "We noticed your stress levels are elevated" ❌
            - "Your stress monitor detected..." ❌
            - "You seem stressed right now" ❌
            """
        } else {
            // CALM STATE - Positive reinforcement with forward-looking micro-suggestion
            prompt = """
            You're a mindfulness coach. A user just returned to calm after being stressed.
            
            Context:
            - Time of day: \(timeOfDay):00 (\(timeOfDayContext(hour: timeOfDay)))
            - Recovery time: \(timestamp.formatted())
            
            Generate a SHORT acknowledgment (MAX 25 words) that:
            1. NO generic "great job" or "you're calm now"
            2. Subtle acknowledgment + ONE forward-looking micro-habit suggestion
            3. Casual, friendly tone
            4. NO markdown, NO emojis, NO asterisks
            
            Good examples:
            - "Nice. When you feel tension building again, try that breathing technique early."
            - "Solid recovery. Jot down what helped—might work next time too."
            - "You're back on track. Remember: small resets work better than powering through."
            
            Bad examples:
            - "Great to see you're feeling calm!" ❌
            - "Stress level normalized" ❌
            - "Excellent! Keep it up!" ❌
            """
        }
        
        do {
            let message = try await callGemini(prompt: prompt)
            // Strip any markdown formatting that might slip through
            return message.replacingOccurrences(of: "**", with: "")
                         .replacingOccurrences(of: "##", with: "")
                         .replacingOccurrences(of: "*", with: "")
                         .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            print("[AIDecisionProvider] Error: \(error)")
            // Fallback message - also improved
            if state == 1 {
                return "Quick reset: Breathe in for 4, hold for 4, out for 6. Do it three times."
            } else {
                return "Nice recovery. Next time, catch the tension early—works better."
            }
        }
    }
    
    /// Time of day context helper
    private func timeOfDayContext(hour: Int) -> String {
        switch hour {
        case 5..<9: return "early morning"
        case 9..<12: return "mid-morning"
        case 12..<14: return "lunch time"
        case 14..<17: return "afternoon"
        case 17..<20: return "evening"
        case 20..<23: return "night"
        default: return "late night"
        }
    }
    
    /// Sync wrapper for backward compatibility
    func message(for event: StressEvent) -> String {
        let state = event.stressState
        if state == 1 {
            return "Quick reset: Breathe in for 4, hold for 4, out for 6. Three times."
        } else {
            return "Nice recovery. Catch the tension early next time—works better."
        }
    }
    
    private func callGemini(prompt: String) async throws -> String {
        let url = URL(string: "\(endpoint)?key=\(apiKey)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = [
            "contents": [
                [
                    "parts": [
                        ["text": prompt]
                    ]
                ]
            ],
            "generationConfig": [
                "temperature": 0.9,
                "maxOutputTokens": 100
            ]
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw NSError(domain: "GeminiAPI", code: -1, userInfo: [NSLocalizedDescriptionKey: "HTTP error"])
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        if let candidates = json?["candidates"] as? [[String: Any]],
           let firstCandidate = candidates.first,
           let content = firstCandidate["content"] as? [String: Any],
           let parts = content["parts"] as? [[String: Any]],
           let text = parts.first?["text"] as? String {
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        throw NSError(domain: "GeminiAPI", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
    }
}
