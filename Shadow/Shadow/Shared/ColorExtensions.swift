import SwiftUI

extension Color {
    init?(hex: String) {
        var hexSanitized = hex.trimmingCharacters(in: .whitespacesAndNewlines)
        hexSanitized = hexSanitized.replacingOccurrences(of: "#", with: "")
        
        var rgb: UInt64 = 0
        
        var r: CGFloat = 0.0
        var g: CGFloat = 0.0
        var b: CGFloat = 0.0
        var a: CGFloat = 1.0
        
        let length = hexSanitized.count
        
        guard Scanner(string: hexSanitized).scanHexInt64(&rgb) else { return nil }
        
        if length == 6 {
            r = CGFloat((rgb & 0xFF0000) >> 16) / 255.0
            g = CGFloat((rgb & 0x00FF00) >> 8) / 255.0
            b = CGFloat(rgb & 0x0000FF) / 255.0
        } else if length == 8 {
            r = CGFloat((rgb & 0xFF000000) >> 24) / 255.0
            g = CGFloat((rgb & 0x00FF0000) >> 16) / 255.0
            b = CGFloat((rgb & 0x0000FF00) >> 8) / 255.0
            a = CGFloat(rgb & 0x000000FF) / 255.0
        } else {
            return nil
        }
        
        self.init(red: r, green: g, blue: b, opacity: a)
    }
    
    // MARK: - Shadow Wellness Theme
    
    // Primary Brand Colors - Calming Teal/Turquoise
    static let shadowPrimary = Color(red: 0.26, green: 0.71, blue: 0.69) // #43B5B0 - Calm teal
    static let shadowPrimaryLight = Color(red: 0.52, green: 0.82, blue: 0.80) // #85D1CC - Light teal
    static let shadowPrimaryDark = Color(red: 0.18, green: 0.56, blue: 0.54) // #2E8F8A - Deep teal
    
    // Secondary Colors - Warm Coral/Peach
    static let shadowSecondary = Color(red: 1.0, green: 0.71, blue: 0.62) // #FFB59E - Soft coral
    static let shadowSecondaryLight = Color(red: 1.0, green: 0.85, blue: 0.80) // #FFD9CC - Light peach
    static let shadowSecondaryDark = Color(red: 0.96, green: 0.55, blue: 0.42) // #F58C6B - Deep coral
    
    // Accent Colors - Soft Lavender/Purple
    static let shadowAccent = Color(red: 0.71, green: 0.65, blue: 0.90) // #B5A6E5 - Soft lavender
    static let shadowAccentLight = Color(red: 0.85, green: 0.82, blue: 0.95) // #D9D1F2 - Light lavender
    static let shadowAccentDark = Color(red: 0.56, green: 0.48, blue: 0.78) // #8F7AC7 - Deep purple
    
    // Background Colors - Soft & Clean
    static let shadowBackground = Color(red: 0.98, green: 0.98, blue: 0.99) // #FAFAFC - Almost white
    static let shadowBackgroundSecondary = Color(red: 0.95, green: 0.96, blue: 0.97) // #F2F4F7 - Light gray
    static let shadowSurface = Color.white
    
    // Gradient Backgrounds
    static let shadowGradientStart = Color(red: 0.96, green: 0.98, blue: 0.99) // #F5F9FC
    static let shadowGradientEnd = Color(red: 0.98, green: 0.95, blue: 0.98) // #FAF2F9
    
    // Text Colors
    static let shadowTextPrimary = Color(red: 0.15, green: 0.18, blue: 0.25) // #262D40 - Deep navy
    static let shadowTextSecondary = Color(red: 0.45, green: 0.50, blue: 0.58) // #737F94 - Medium gray
    static let shadowTextTertiary = Color(red: 0.65, green: 0.68, blue: 0.74) // #A6ADB9 - Light gray
    
    // State Colors - Wellness Focused
    static let shadowSuccess = Color(red: 0.40, green: 0.76, blue: 0.65) // #66C2A5 - Calm green
    static let shadowSuccessLight = Color(red: 0.80, green: 0.93, blue: 0.88) // #CCEEE0 - Light green
    
    static let shadowWarning = Color(red: 0.98, green: 0.76, blue: 0.45) // #FAC273 - Warm amber
    static let shadowWarningLight = Color(red: 1.0, green: 0.92, blue: 0.82) // #FFEAD1 - Light amber
    
    static let shadowError = Color(red: 0.95, green: 0.58, blue: 0.58) // #F29494 - Soft red
    static let shadowErrorLight = Color(red: 0.99, green: 0.88, blue: 0.88) // #FCE0E0 - Light red
    
    static let shadowInfo = Color(red: 0.53, green: 0.74, blue: 0.94) // #87BCF0 - Soft blue
    static let shadowInfoLight = Color(red: 0.88, green: 0.94, blue: 0.99) // #E0F0FC - Light blue
    
    // Stress Level Colors - Gradient Scale
    static let shadowStressLow = Color(red: 0.66, green: 0.89, blue: 0.82) // #A8E4D1 - Very calm
    static let shadowStressMedium = Color(red: 0.98, green: 0.82, blue: 0.53) // #FAD187 - Gentle alert
    static let shadowStressHigh = Color(red: 0.98, green: 0.69, blue: 0.62) // #FAB09E - Warm attention
    
    // Shadow/Elevation Colors
    static let shadowElevation1 = Color.black.opacity(0.05)
    static let shadowElevation2 = Color.black.opacity(0.08)
    static let shadowElevation3 = Color.black.opacity(0.12)
    
    // Border Colors
    static let shadowBorder = Color(red: 0.90, green: 0.92, blue: 0.94) // #E5EBF0
    static let shadowBorderLight = Color(red: 0.95, green: 0.96, blue: 0.97) // #F2F4F7
    
    // MARK: - Gradient Helpers
    static func shadowPrimaryGradient() -> LinearGradient {
        LinearGradient(
            gradient: Gradient(colors: [shadowPrimary, shadowPrimaryLight]),
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
    
    static func shadowWellnessGradient() -> LinearGradient {
        LinearGradient(
            gradient: Gradient(colors: [shadowGradientStart, shadowGradientEnd]),
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
    
    static func shadowAccentGradient() -> LinearGradient {
        LinearGradient(
            gradient: Gradient(colors: [shadowAccent, shadowAccentLight]),
            startPoint: .leading,
            endPoint: .trailing
        )
    }
    
    static func shadowCalmGradient() -> LinearGradient {
        LinearGradient(
            gradient: Gradient(colors: [
                Color(red: 0.85, green: 0.95, blue: 0.98), // #D9F2FA - Light cyan
                Color(red: 0.92, green: 0.94, blue: 0.99)  // #EBF0FC - Light blue
            ]),
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}
