CXX ?= clang++
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -Iinclude -ffp-contract=off
LDLIBS =
TARGET = HydraMa2D

# Optional CGNS mesh input (the CMake build finds CGNS by itself):
#   make CGNS=1                                   CGNS in /usr
#   make CGNS=1 CGNS_PREFIX=/opt/cgns             headers in $(CGNS_PREFIX)/include
#   make CGNS=1 CGNS_LIBDIR=/usr/lib/x86_64-linux-gnu
# A static libcgns also needs HDF5: add e.g. CGNS_LIBS="-lcgns -lhdf5".
# Switching CGNS on or off needs a "make clean" (the objects are compiled differently).
ifeq ($(CGNS),1)
CGNS_PREFIX ?= /usr
CGNS_LIBDIR ?= $(CGNS_PREFIX)/lib
CGNS_LIBS ?= -lcgns
CXXFLAGS += -DHYDRAMA2D_HAVE_CGNS=1 -isystem $(CGNS_PREFIX)/include
# -rpath: the linker also looks there for the libraries libcgns needs (HDF5),
# and the program finds libcgns at run time.
LDLIBS += -L$(CGNS_LIBDIR) -Wl,-rpath,$(CGNS_LIBDIR) $(CGNS_LIBS)
endif

SRCS = $(wildcard src/*.cpp src/*/*.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o src/*/*.o
