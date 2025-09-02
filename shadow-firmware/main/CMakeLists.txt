idf_component_register(SRCS "main_realtime.c"
                    INCLUDE_DIRS "."
                    REQUIRES freertos sensor_buffer feature_extractor ml_model driver
                    PRIV_REQUIRES esp_timer esp_system)