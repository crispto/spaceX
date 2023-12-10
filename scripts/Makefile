BUILD := build
SRC := $(wildcard *.cpp)
OBJS := $(SRC:.cpp=.o)
# print SRC
DEPS := gstreamer-1.0

CFLAGS += $(shell pkg-config --cflags $(DEPS))
LIBS += $(shell pkg-config --libs $(DEPS))

$(BUILD)/%.o: %.cpp
	$(CXX) -c -o $@ $(CFLAGS) $<

main: $(addprefix $(BUILD)/, $(OBJS)) Makefile
	$(CXX) -o $(BUILD)/$@ $(addprefix $(BUILD)/, $(OBJS)) $(LIBS)
