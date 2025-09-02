idf_component_register(SRCS "event_log.c"
                       INCLUDE_DIRS "include"
                       REQUIRES freertos esp_timer stress_fsm)
