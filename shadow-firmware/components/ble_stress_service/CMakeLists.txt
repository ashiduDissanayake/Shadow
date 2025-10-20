idf_component_register(SRCS "ble_stress_service.c" "ble_pairing.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos esp_timer nvs_flash bt stress_fsm event_log mbedtls time_sync)
