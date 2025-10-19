idf_component_register(
    SRCS "main_realtime.c"
    INCLUDE_DIRS "."
    REQUIRES 
        freertos 
        sensor_buffer 
        feature_extractor 
        ml_model 
        driver 
        stress_fsm 
        event_log 
        ble_stress_service
        signal_preprocessor  # Added: Phase 2 signal preprocessing
        cnn_inference        # Added: Phase 3 CNN inference
        display_manager      # Added: TFT display and QR code
        esp_adc          # Added: Required for ESP-IDF v5.5 ADC
        esp_timer        # Moved from PRIV_REQUIRES
        esp_system       # Moved from PRIV_REQUIRES
        log              # Added: Required for ESP_LOG functions
        esp_common       # Added: Required for esp_err_to_name()
    PRIV_REQUIRES 
        # Remove esp_timer and esp_system from here since they're now in REQUIRES
)