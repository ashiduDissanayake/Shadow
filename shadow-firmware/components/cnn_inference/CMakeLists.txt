idf_component_register(
    SRCS "cnn_inference.cpp" "stress_model_data.c"
    INCLUDE_DIRS "include"
    REQUIRES signal_preprocessor esp_timer log esp-tflite-micro
)

# Force standard TFLite kernels (disable ESP-NN optimized kernels)
target_compile_definitions(${COMPONENT_LIB} PRIVATE 
    ESP_NN_DISABLE=1
)
