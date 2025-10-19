idf_component_register(
    SRCS "src/qrcode.c"
    INCLUDE_DIRS "src"
)

# Disable warnings for third-party QR code library
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-unknown-pragmas
    -Wno-type-limits
)
