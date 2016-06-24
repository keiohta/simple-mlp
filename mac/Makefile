TARGET = simple-mlp
CXXFLAGS = -O4 -g -Wall -std=c++0x
 
SRCS = $(shell ls ../src/*.cpp)
OBJS = $(SRCS:.cpp=.o)
HEADS = $(shell ls ../src/*.h)
LIBS = -lpthread
 
$(TARGET): $(OBJS) $(HEADS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)
 
all: $(TARGET)
 
run: all
	./$(TARGET)

depend:
	$(CXX) -MM -MG $(SRCS) > Makefile.depend
	cat Makefile.depend

clean:
	rm -f $(OBJS) $(TARGET) *~ ¥#*¥#
	rm -f *.o
