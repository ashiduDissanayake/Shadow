# 🎨 Shadow Wellness Theme Update

## Overview
The Shadow app has been completely redesigned with a calming, wellness-focused color theme that's more appropriate for a health and stress monitoring application. The dark, stark theme has been replaced with soft, soothing colors that promote a sense of calm and well-being.

## New Color Palette

### Primary Colors - Calming Teal/Turquoise
- **Primary**: `#43B5B0` - Calm teal (associated with healing and tranquility)
- **Primary Light**: `#85D1CC` - Light teal
- **Primary Dark**: `#2E8F8A` - Deep teal

### Secondary Colors - Warm Coral/Peach
- **Secondary**: `#FFB59E` - Soft coral (encouraging and gentle)
- **Secondary Light**: `#FFD9CC` - Light peach
- **Secondary Dark**: `#F58C6B` - Deep coral

### Accent Colors - Soft Lavender/Purple
- **Accent**: `#B5A6E5` - Soft lavender (mindfulness and wellness)
- **Accent Light**: `#D9D1F2` - Light lavender
- **Accent Dark**: `#8F7AC7` - Deep purple

### Background Colors
- **Background**: `#FAFAFC` - Almost white (clean and calm)
- **Background Secondary**: `#F2F4F7` - Light gray
- **Surface**: `#FFFFFF` - Pure white

### Text Colors
- **Text Primary**: `#262D40` - Deep navy (excellent readability)
- **Text Secondary**: `#737F94` - Medium gray
- **Text Tertiary**: `#A6ADB9` - Light gray

### State Colors - Wellness Focused
- **Success**: `#66C2A5` - Calm green (balanced state)
- **Warning**: `#FAC273` - Warm amber (gentle alert)
- **Error**: `#F29494` - Soft red (non-alarming)
- **Info**: `#87BCF0` - Soft blue (informative)

### Stress Level Colors
- **Low Stress**: `#A8E4D1` - Very calm green
- **Medium Stress**: `#FAD187` - Gentle alert amber
- **High Stress**: `#FAB09E` - Warm attention coral

## Design Philosophy

### 1. **Calming & Supportive**
   - Soft, muted colors reduce anxiety
   - Teal primary color associated with healing and calmness
   - Warm coral for encouragement without alarm

### 2. **Professional Health Focus**
   - Clean white backgrounds for clarity
   - Professional navy text for readability
   - Subtle gradients for depth without distraction

### 3. **Stress-Aware Color Coding**
   - Gentle color transitions for stress states
   - No harsh reds - using soft coral instead
   - Success states in calming green tones

### 4. **Accessibility**
   - High contrast text (navy on white)
   - Clear visual hierarchy
   - Soft shadows for depth perception
   - WCAG AA compliant color combinations

## Updated Components

### ✅ Authentication
- **LoginView**: Light, welcoming gradient background with wellness colors
- Feature pills updated with new theme

### ✅ Dashboard
- **ShadowDashboardView**: Soft wellness gradient background
- Cards with white surfaces and subtle shadows
- Status indicators using wellness colors
- Graph section with calming accent colors

### ✅ Navigation
- **ShadowAppNavBar**: White surface with soft shadows
- Teal primary accents
- Clean, modern appearance

### ✅ Calendar
- **CalendarMainView**: Wellness gradient background
- White cards with subtle elevation
- Teal primary for actions
- Soft accent colors for stats

### ✅ Profile
- **ProfileView**: Consistent wellness theme
- Soft backgrounds for forms
- Calming profile avatar gradient
- Clear action hierarchy

### ✅ Components
- **FeaturePill**: White backgrounds with colored icons
- **RecentStressEventRow**: Soft backgrounds, wellness state colors
- **ShadowButtonStyle**: Wellness color variants

## Gradient Helpers

The `ColorExtensions.swift` file now includes helpful gradient generators:

```swift
Color.shadowPrimaryGradient()      // Teal gradient
Color.shadowWellnessGradient()     // Main background gradient
Color.shadowAccentGradient()       // Lavender gradient
Color.shadowCalmGradient()         // Cyan-blue calm gradient
```

## Usage Examples

### Using the new colors in views:
```swift
// Background
.background(Color.shadowWellnessGradient())

// Primary button
.background(Color.shadowPrimaryGradient())

// Success state
.foregroundColor(.shadowSuccess)

// Cards
.background(Color.shadowSurface)
.shadow(color: Color.shadowElevation2, radius: 8, x: 0, y: 2)

// Text
.foregroundColor(.shadowTextPrimary)    // Main text
.foregroundColor(.shadowTextSecondary)  // Secondary text
.foregroundColor(.shadowTextTertiary)   // Tertiary text
```

## Benefits of the New Theme

1. **Reduced Eye Strain**: Light backgrounds are easier on the eyes for extended use
2. **Calming Effect**: Teal and soft colors promote relaxation
3. **Professional Appearance**: Clean, modern design suitable for health apps
4. **Better Information Hierarchy**: Clear visual distinction between elements
5. **Stress-Appropriate**: Colors don't add to user's stress levels
6. **Accessibility**: Better contrast for readability
7. **Modern & Fresh**: Contemporary design language
8. **Wellness-Focused**: Colors chosen specifically for health and wellness context

## Color Psychology

- **Teal/Turquoise**: Healing, emotional balance, tranquility, clarity
- **Coral/Peach**: Warmth, encouragement, gentle energy
- **Lavender**: Mindfulness, calm, wellness, mental clarity
- **Soft Green**: Balance, growth, harmony, health
- **Warm Amber**: Gentle attention, awareness without alarm

## Next Steps

The theme is now fully integrated across all major views. Future considerations:
- Custom chart colors for stress visualization
- Dark mode variant with wellness-focused dark colors
- Accessibility settings for color blindness
- Animation transitions that reinforce the calm theme

---

**Last Updated**: October 23, 2025  
**Theme Version**: 1.0 - Wellness Edition
