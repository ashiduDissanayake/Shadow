idf_component_register(SRCS "signal_preprocessor.c"
                      INCLUDE_DIRS "include"
                      REQUIRES sensor_buffer)

# Test file is excluded from main build
# To run tests, use: idf.py test
