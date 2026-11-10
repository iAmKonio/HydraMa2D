CXX ?= clang++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude
TARGET = HydraMa2D

SRCS = $(wildcard src/*.cpp src/linalg/*.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o src/linalg/*.o
