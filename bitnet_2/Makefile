CC = gcc
CFLAGS = -Wall -O2 -std=c99
LDFLAGS = -lm

# Source files
SRCS = bitnet.c
OBJS = $(SRCS:.c=.o)

# Target executable
TARGET = bitnet

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET) BitNet.txt

# Run the program
run: $(TARGET)
	./$(TARGET)

# Dependencies
.PHONY: all clean run
