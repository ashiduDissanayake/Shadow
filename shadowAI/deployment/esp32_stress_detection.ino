/*
 * ESP32-S3 BVP-Based Stress Detection Firmware
 * 
 * This firmware implements real-time stress detection using wrist-based
 * Photoplethysmography (BVP) signals and the H-CNN model.
 * 
 * Hardware Requirements:
 * - ESP32-S3 development board
 * - MAX30102 or similar PPG sensor
 * - I2C display (optional)
 * - LED indicators
 * 
 * Features:
 * - Real-time BVP signal acquisition
 * - Signal preprocessing and feature extraction
 * - H-CNN model inference
 * - Stress level classification
 * - Data logging and communication
 */

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // Auto-generated from model_optimization.py

// Hardware configuration
#define PPG_SDA_PIN 21
#define PPG_SCL_PIN 22
#define PPG_INT_PIN 23
#define LED_STRESS_PIN 2
#define LED_BASELINE_PIN 4
#define LED_AMUSEMENT_PIN 5
#define BUZZER_PIN 18

// PPG sensor configuration
#define PPG_SAMPLING_RATE 64  // Hz
#define SEGMENT_LENGTH 3840   // 60 seconds * 64 Hz
#define FEATURE_COUNT 9
#define NUM_CLASSES 3

// Model configuration
#define TENSOR_ARENA_SIZE 100000  // Adjust based on model size

// Global variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_segment = nullptr;
TfLiteTensor* input_features = nullptr;
TfLiteTensor* output = nullptr;

// Data buffers
float bvp_buffer[SEGMENT_LENGTH];
float feature_buffer[FEATURE_COUNT];
int buffer_index = 0;
bool buffer_full = false;

// Classification results
enum StressLevel {
  BASELINE = 0,
  STRESS = 1,
  AMUSEMENT = 2
};

StressLevel current_stress_level = BASELINE;
float confidence_scores[NUM_CLASSES];

// Timing variables
unsigned long last_sample_time = 0;
unsigned long last_inference_time = 0;
const unsigned long SAMPLE_INTERVAL = 1000000 / PPG_SAMPLING_RATE;  // microseconds
const unsigned long INFERENCE_INTERVAL = 60000000;  // 60 seconds

// PPG sensor functions (placeholder - implement based on your sensor)
class PPGSensor {
private:
  uint8_t address;
  
public:
  PPGSensor(uint8_t addr = 0x57) : address(addr) {}
  
  bool begin() {
    Wire.begin(PPG_SDA_PIN, PPG_SCL_PIN);
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
  }
  
  float readBVP() {
    // Placeholder implementation
    // Replace with actual PPG sensor reading code
    Wire.beginTransmission(address);
    Wire.write(0x00);  // Register address for data
    Wire.endTransmission();
    
    Wire.requestFrom(address, 3);
    if (Wire.available() >= 3) {
      uint8_t msb = Wire.read();
      uint8_t lsb = Wire.read();
      uint8_t status = Wire.read();
      
      int16_t raw_value = (msb << 8) | lsb;
      return (float)raw_value / 32768.0;  // Normalize to [-1, 1]
    }
    
    return 0.0;
  }
  
  bool dataReady() {
    // Check if new data is available
    Wire.beginTransmission(address);
    Wire.write(0x01);  // Status register
    Wire.endTransmission();
    
    Wire.requestFrom(address, 1);
    if (Wire.available()) {
      uint8_t status = Wire.read();
      return (status & 0x01) != 0;
    }
    
    return false;
  }
};

PPGSensor ppg;

// Signal processing functions
class SignalProcessor {
private:
  // Butterworth filter coefficients (pre-computed for 0.7-3.7 Hz bandpass)
  static const int FILTER_ORDER = 3;
  float filter_a[FILTER_ORDER + 1] = {1.0, -2.374, 1.929, -0.555};
  float filter_b[FILTER_ORDER + 1] = {0.001, 0.003, 0.003, 0.001};
  float filter_x[FILTER_ORDER + 1] = {0.0};
  float filter_y[FILTER_ORDER + 1] = {0.0};
  
public:
  float filterSignal(float input) {
    // Shift input history
    for (int i = FILTER_ORDER; i > 0; i--) {
      filter_x[i] = filter_x[i-1];
      filter_y[i] = filter_y[i-1];
    }
    filter_x[0] = input;
    
    // Apply filter
    float output = 0.0;
    for (int i = 0; i <= FILTER_ORDER; i++) {
      output += filter_b[i] * filter_x[i];
    }
    for (int i = 1; i <= FILTER_ORDER; i++) {
      output -= filter_a[i] * filter_y[i];
    }
    
    filter_y[0] = output;
    return output;
  }
  
  void extractFeatures(float* segment, float* features) {
    // Calculate basic statistical features
    float mean_val = 0.0, std_val = 0.0, min_val = segment[0], max_val = segment[0];
    
    // Mean, min, max
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
      mean_val += segment[i];
      if (segment[i] < min_val) min_val = segment[i];
      if (segment[i] > max_val) max_val = segment[i];
    }
    mean_val /= SEGMENT_LENGTH;
    
    // Standard deviation
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
      std_val += (segment[i] - mean_val) * (segment[i] - mean_val);
    }
    std_val = sqrt(std_val / SEGMENT_LENGTH);
    
    // Peak detection for HRV features
    int peak_count = 0;
    float rr_intervals[100];  // Store RR intervals
    int rr_count = 0;
    
    for (int i = 1; i < SEGMENT_LENGTH - 1; i++) {
      if (segment[i] > segment[i-1] && segment[i] > segment[i+1] && 
          segment[i] > mean_val + 0.5 * std_val) {
        peak_count++;
        if (peak_count > 1) {
          // Calculate RR interval (simplified)
          float rr_interval = (float)(i - (i-1)) / PPG_SAMPLING_RATE * 1000;  // ms
          if (rr_count < 100) {
            rr_intervals[rr_count++] = rr_interval;
          }
        }
      }
    }
    
    // Calculate HRV features
    float mean_rr = 0.0, std_rr = 0.0, rmssd = 0.0;
    int nn50 = 0;
    
    if (rr_count > 1) {
      // Mean RR
      for (int i = 0; i < rr_count; i++) {
        mean_rr += rr_intervals[i];
      }
      mean_rr /= rr_count;
      
      // Standard deviation RR
      for (int i = 0; i < rr_count; i++) {
        std_rr += (rr_intervals[i] - mean_rr) * (rr_intervals[i] - mean_rr);
      }
      std_rr = sqrt(std_rr / rr_count);
      
      // RMSSD
      for (int i = 1; i < rr_count; i++) {
        float diff = rr_intervals[i] - rr_intervals[i-1];
        rmssd += diff * diff;
      }
      rmssd = sqrt(rmssd / (rr_count - 1));
      
      // NN50
      for (int i = 1; i < rr_count; i++) {
        if (abs(rr_intervals[i] - rr_intervals[i-1]) > 50) {
          nn50++;
        }
      }
    }
    
    float pnn50 = (rr_count > 0) ? (float)nn50 / rr_count * 100 : 0.0;
    float heart_rate = (mean_rr > 0) ? 60000.0 / mean_rr : 60.0;
    
    // Fill feature array
    features[0] = mean_val;
    features[1] = std_val;
    features[2] = min_val;
    features[3] = max_val;
    features[4] = heart_rate;
    features[5] = mean_rr;
    features[6] = std_rr;
    features[7] = rmssd;
    features[8] = pnn50;
  }
};

SignalProcessor signal_processor;

// TensorFlow Lite setup
bool setupTensorFlowLite() {
  // Map the model into a usable data structure
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    return false;
  }
  
  // Pull in all the operation implementations we need
  static tflite::AllOpsResolver resolver;
  
  // Create an area of memory to use for input, output, and intermediate arrays
  static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return false;
  }
  
  // Obtain pointers to the model's input and output tensors
  input_segment = interpreter->input(0);
  input_features = interpreter->input(1);
  output = interpreter->output(0);
  
  return true;
}

// Inference function
StressLevel runInference() {
  // Prepare input data
  // Normalize segment (z-score normalization)
  float segment_mean = 0.0, segment_std = 0.0;
  for (int i = 0; i < SEGMENT_LENGTH; i++) {
    segment_mean += bvp_buffer[i];
  }
  segment_mean /= SEGMENT_LENGTH;
  
  for (int i = 0; i < SEGMENT_LENGTH; i++) {
    segment_std += (bvp_buffer[i] - segment_mean) * (bvp_buffer[i] - segment_mean);
  }
  segment_std = sqrt(segment_std / SEGMENT_LENGTH);
  
  // Fill input tensors
  for (int i = 0; i < SEGMENT_LENGTH; i++) {
    float normalized_value = (segment_std > 0) ? (bvp_buffer[i] - segment_mean) / segment_std : 0.0;
    input_segment->data.f[i] = normalized_value;
  }
  
  // Normalize features (using pre-computed scaler parameters)
  // In a real implementation, you would load the actual scaler parameters
  float feature_mean[FEATURE_COUNT] = {0.0, 0.0, 0.0, 0.0, 70.0, 1000.0, 50.0, 30.0, 20.0};
  float feature_scale[FEATURE_COUNT] = {1.0, 1.0, 1.0, 1.0, 20.0, 200.0, 30.0, 20.0, 30.0};
  
  for (int i = 0; i < FEATURE_COUNT; i++) {
    float normalized_feature = (feature_buffer[i] - feature_mean[i]) / feature_scale[i];
    input_features->data.f[i] = normalized_feature;
  }
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed!");
    return BASELINE;
  }
  
  // Get results
  int max_index = 0;
  float max_score = output->data.f[0];
  
  for (int i = 0; i < NUM_CLASSES; i++) {
    confidence_scores[i] = output->data.f[i];
    if (output->data.f[i] > max_score) {
      max_score = output->data.f[i];
      max_index = i;
    }
  }
  
  return (StressLevel)max_index;
}

// LED control functions
void updateLEDs(StressLevel level) {
  // Turn off all LEDs
  digitalWrite(LED_BASELINE_PIN, LOW);
  digitalWrite(LED_STRESS_PIN, LOW);
  digitalWrite(LED_AMUSEMENT_PIN, LOW);
  
  // Turn on appropriate LED
  switch (level) {
    case BASELINE:
      digitalWrite(LED_BASELINE_PIN, HIGH);
      break;
    case STRESS:
      digitalWrite(LED_STRESS_PIN, HIGH);
      // Optional: trigger buzzer for stress
      if (confidence_scores[STRESS] > 0.8) {
        tone(BUZZER_PIN, 1000, 500);
      }
      break;
    case AMUSEMENT:
      digitalWrite(LED_AMUSEMENT_PIN, HIGH);
      break;
  }
}

// Serial output functions
void printResults(StressLevel level) {
  Serial.print("Stress Level: ");
  switch (level) {
    case BASELINE:
      Serial.print("BASELINE");
      break;
    case STRESS:
      Serial.print("STRESS");
      break;
    case AMUSEMENT:
      Serial.print("AMUSEMENT");
      break;
  }
  
  Serial.print(" | Confidence: [");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(confidence_scores[i], 3);
    if (i < NUM_CLASSES - 1) Serial.print(", ");
  }
  Serial.println("]");
}

// Setup function
void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.println("ESP32-S3 BVP Stress Detection System");
  
  // Initialize GPIO pins
  pinMode(LED_BASELINE_PIN, OUTPUT);
  pinMode(LED_STRESS_PIN, OUTPUT);
  pinMode(LED_AMUSEMENT_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Initialize PPG sensor
  if (!ppg.begin()) {
    Serial.println("Failed to initialize PPG sensor!");
    while (1);
  }
  Serial.println("PPG sensor initialized");
  
  // Initialize TensorFlow Lite
  if (!setupTensorFlowLite()) {
    Serial.println("Failed to initialize TensorFlow Lite!");
    while (1);
  }
  Serial.println("TensorFlow Lite initialized");
  
  // Initialize data buffer
  memset(bvp_buffer, 0, sizeof(bvp_buffer));
  buffer_index = 0;
  buffer_full = false;
  
  Serial.println("System ready for stress detection");
}

// Main loop
void loop() {
  unsigned long current_time = micros();
  
  // Sample BVP data at specified rate
  if (current_time - last_sample_time >= SAMPLE_INTERVAL) {
    if (ppg.dataReady()) {
      float bvp_value = ppg.readBVP();
      
      // Apply signal processing
      float filtered_value = signal_processor.filterSignal(bvp_value);
      
      // Store in buffer
      bvp_buffer[buffer_index] = filtered_value;
      buffer_index++;
      
      if (buffer_index >= SEGMENT_LENGTH) {
        buffer_index = 0;
        buffer_full = true;
      }
      
      last_sample_time = current_time;
    }
  }
  
  // Run inference every 60 seconds when buffer is full
  if (buffer_full && (current_time - last_inference_time >= INFERENCE_INTERVAL)) {
    // Extract features from the current segment
    signal_processor.extractFeatures(bvp_buffer, feature_buffer);
    
    // Run inference
    StressLevel new_stress_level = runInference();
    
    // Update current stress level
    current_stress_level = new_stress_level;
    
    // Update LEDs
    updateLEDs(current_stress_level);
    
    // Print results
    printResults(current_stress_level);
    
    // Reset timing
    last_inference_time = current_time;
    
    // Optional: Add data logging or communication here
  }
  
  // Small delay to prevent watchdog issues
  delay(1);
}
