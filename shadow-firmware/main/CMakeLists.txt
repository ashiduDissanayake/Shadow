idf_component_register(SRCS "main_realtime.c"
                    INCLUDE_DIRS "."
                    REQUIRES freertos sensor_buffer feature_extractor ml_model driver stress_fsm event_log ble_stress_service
                    PRIV_REQUIRES esp_timer esp_system)