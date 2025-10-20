idf_component_register(SRCS "calibration.c" "signal_preprocessor.c"
                      INCLUDE_DIRS "include"
                      REQUIRES sensor_buffer nvs_flash)

# Test file is excluded from main build
# To run tests, use: idf.py test
