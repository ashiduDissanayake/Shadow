# ESP32 Multi-Sensor Buffer Makefile

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -g
INCLUDES = -I.
LIBS = -lm

# Source files
SOURCES = sensor_buffer.c feature_extractor.c test_buffer.c test_features.c test_complete.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = test_buffer
FEATURE_TARGET = test_features
COMPLETE_TARGET = test_complete

# Default target
all: $(TARGET) $(FEATURE_TARGET) $(COMPLETE_TARGET)

# Build target
$(TARGET): sensor_buffer.o test_buffer.o
	$(CC) sensor_buffer.o test_buffer.o -o $(TARGET) $(LIBS)
	@echo "✅ Build completed: $(TARGET)"

# Build feature target
$(FEATURE_TARGET): sensor_buffer.o feature_extractor.o test_features.o
	$(CC) sensor_buffer.o feature_extractor.o test_features.o -o $(FEATURE_TARGET) $(LIBS)
	@echo "✅ Build completed: $(FEATURE_TARGET)"

# Build complete target
$(COMPLETE_TARGET): sensor_buffer.o feature_extractor.o test_complete.o
	$(CC) sensor_buffer.o feature_extractor.o test_complete.o -o $(COMPLETE_TARGET) $(LIBS)
	@echo "✅ Build completed: $(COMPLETE_TARGET)"

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
	@echo "🔧 Compiled: $<"

# Run tests
test: $(TARGET)
	@echo "🧪 Running buffer tests..."
	./$(TARGET)

# Run feature tests
test-features: $(FEATURE_TARGET)
	@echo "🧪 Running feature extraction tests..."
	./$(FEATURE_TARGET)

# Run complete system test
test-complete: $(COMPLETE_TARGET)
	@echo "🧪 Running complete system test..."
	./$(COMPLETE_TARGET)

# Run all tests
test-all: test test-complete

# Memory analysis
memory: $(TARGET)
	@echo "📊 Memory analysis:"
	@size $(TARGET)
	@echo "📈 Static memory usage (approximate):"
	@grep -E "BVP_BUFFER_SIZE|ACC_BUFFER_SIZE|EDA_BUFFER_SIZE|TEMP_BUFFER_SIZE" sensor_buffer.h | \
	 sed 's/[^0-9]*\([0-9]*\).*/\1/' | \
	 awk 'BEGIN{total=0} {total+=$$1*4} END{printf "  Total buffer memory: %d bytes (%.1f KB)\n", total, total/1024}'

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) $(FEATURE_TARGET) $(COMPLETE_TARGET)
	@echo "🧹 Cleaned build files"

# Debug build
debug: CFLAGS += -DDEBUG -O0
debug: $(TARGET) $(COMPLETE_TARGET)
	@echo "🐛 Debug build completed"

# Profile build
profile: CFLAGS += -pg
profile: $(TARGET) $(COMPLETE_TARGET)
	@echo "📊 Profile build completed"

# Static analysis
analyze:
	@echo "🔍 Running static analysis..."
	@which cppcheck > /dev/null 2>&1 && cppcheck --enable=all --std=c99 sensor_buffer.c feature_extractor.c || echo "❌ cppcheck not found"
	@which splint > /dev/null 2>&1 && splint sensor_buffer.c feature_extractor.c || echo "❌ splint not found"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@which doxygen > /dev/null 2>&1 && doxygen || echo "❌ doxygen not found"

# Help
help:
	@echo "🚀 ESP32 Multi-Sensor Buffer Build System"
	@echo "==========================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all test programs (default)"
	@echo "  test         - Build and run buffer tests"
	@echo "  test-complete- Build and run complete system test"
	@echo "  test-all     - Run all tests"
	@echo "  memory       - Analyze memory usage"
	@echo "  clean        - Remove build files"
	@echo "  debug        - Build with debug flags"
	@echo "  profile      - Build with profiling support"
	@echo "  analyze      - Run static analysis tools"
	@echo "  docs         - Generate documentation"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make test-complete  # Test complete system"
	@echo "  make test-all       # Test everything"
	@echo "  make memory         # Check memory usage"

.PHONY: all test test-features test-complete test-all memory clean debug profile analyze docs help
