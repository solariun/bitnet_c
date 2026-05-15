# Makefile for Adaline Network (Modernized with BitNet Quantization)
# Author: AI Assistant
# Date: 2026

# Compiler settings
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -pedantic
LDFLAGS = 

# Target executable
TARGET = adelide_modern

# Source files
SRCS = adelide_modern.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Install target (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall target (optional)
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Test target
test: $(TARGET)
	./$(TARGET) > test_output.txt && echo "Test completed. Output in test_output.txt"

# Phony targets
.PHONY: all clean install uninstall run test

# Dependencies (for development)
depend:
	$(CC) -MM $(SRCS) > .depend

-include .depend