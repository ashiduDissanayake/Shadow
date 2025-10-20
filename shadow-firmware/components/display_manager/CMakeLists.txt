idf_component_register(
    SRCS "display_manager.c"
    INCLUDE_DIRS "include" "../qrcode/src"
    REQUIRES driver log esp_common esp_lcd qrcode time_sync
)
