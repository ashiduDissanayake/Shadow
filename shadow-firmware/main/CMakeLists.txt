idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES freertos sensor_buffer feature_extractor ml_model
                    PRIV_REQUIRES esp_timer esp_system)