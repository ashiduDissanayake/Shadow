/*
 * Simple TFT Display Test for LilyGo T-Display S3
 * 
 * This test uses the Arduino TFT_eSPI library to verify:
 * 1. Display hardware works
 * 2. Pin configuration is correct
 * 3. Interface type (SPI vs Parallel)
 * 
 * Based on the clock example - shows a simple digital clock
 */

#include <TFT_eSPI.h> // Graphics and font library for ST7735 driver chip
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI();  // Invoke library, pins defined in User_Setup.h

uint32_t targetTime = 0;       // for next 1 second timeout

byte omm = 99;
bool initial = 1;
byte xcolon = 0;
unsigned int colour = 0;

static uint8_t conv2d(const char* p) {
  uint8_t v = 0;
  if ('0' <= *p && *p <= '9')
    v = *p - '0';
  return 10 * v + *++p - '0';
}

uint8_t hh=conv2d(__TIME__), mm=conv2d(__TIME__+3), ss=conv2d(__TIME__+6);  // Get H, M, S from compile time

void setup(void) {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("=== LilyGo T-Display S3 Test ===");
  Serial.println("Initializing TFT display...");
  
  tft.init();
  tft.setRotation(1);  // Landscape orientation
  tft.fillScreen(TFT_BLACK);
  
  Serial.println("Display initialized successfully!");
  Serial.printf("Display size: %d x %d\n", tft.width(), tft.height());
  
  // Test: Fill screen with colors to verify display works
  Serial.println("Testing colors...");
  tft.fillScreen(TFT_RED);
  delay(500);
  tft.fillScreen(TFT_GREEN);
  delay(500);
  tft.fillScreen(TFT_BLUE);
  delay(500);
  tft.fillScreen(TFT_BLACK);
  
  Serial.println("Color test complete!");

  tft.setTextColor(TFT_YELLOW, TFT_BLACK); // Note: the new fonts do not draw the background colour
  
  // Display title
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.setCursor(8, 52);
  tft.print("SHADOW Display Test");
  
  targetTime = millis() + 1000; 
}

void loop() {
  if (targetTime < millis()) {
    targetTime = millis()+1000;
    ss++;              // Advance second
    if (ss==60) {
      ss=0;
      omm = mm;
      mm++;            // Advance minute
      if(mm>59) {
        mm=0;
        hh++;          // Advance hour
        if (hh>23) {
          hh=0;
        }
      }
    }

    if (ss==0 || initial) {
      initial = 0;
      tft.setTextColor(TFT_GREEN, TFT_BLACK);
      tft.setCursor (8, 80);
      tft.print(__DATE__); // This uses the standard ADAFruit small font

      tft.setTextColor(TFT_BLUE, TFT_BLACK);
      tft.drawCentreString("LilyGo T-Display S3", 160, 100, 2); // Centered text
    }

    // Update digital time
    byte xpos = 6;
    byte ypos = 0;
    if (omm != mm) { // Only redraw every minute to minimise flicker
      // Uncomment ONE of the next 2 lines, using the ghost image demonstrates text overlay as time is drawn over it
      tft.setTextColor(0x39C4, TFT_BLACK);  // Leave a 7 segment ghost image
      // Font 7 is to show a pseudo 7 segment display.
      // Font 7 only contains characters [space] 0 1 2 3 4 5 6 7 8 9 0 : .
      tft.drawString("88:88", xpos, ypos, 7); // Overwrite the text to clear it
      tft.setTextColor(0xFBE0); // Orange
      omm = mm;

      if (hh<10) xpos+= tft.drawChar('0', xpos, ypos, 7);
      xpos+= tft.drawNumber(hh, xpos, ypos, 7);
      xcolon=xpos;
      xpos+= tft.drawChar(':', xpos, ypos, 7);
      if (mm<10) xpos+= tft.drawChar('0', xpos, ypos, 7);
      tft.drawNumber(mm, xpos, ypos, 7);
    }

    if (ss%2) { // Flash the colon
      tft.setTextColor(0x39C4, TFT_BLACK);
      xpos+= tft.drawChar(':', xcolon, ypos, 7);
      tft.setTextColor(0xFBE0, TFT_BLACK);
    }
    else {
      tft.drawChar(':', xcolon, ypos, 7);
      colour = random(0xFFFF);
      // Erase the old text with a rectangle
      tft.fillRect(0, 120, 320, 30, TFT_BLACK);
      tft.setTextColor(colour);
      tft.drawRightString("Colour", 140, 120, 4); // Right justified
      String scolour = String(colour, HEX);
      scolour.toUpperCase();
      char buffer[20];
      scolour.toCharArray(buffer, 20);
      tft.drawString(buffer, 150, 120, 4);
    }
    
    // Print to serial for debugging
    Serial.printf("Time: %02d:%02d:%02d\n", hh, mm, ss);
  }
}
