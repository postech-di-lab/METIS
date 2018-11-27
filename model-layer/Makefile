CXX = g++

LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER

OPT = -std=c++11 -mcmodel=medium -g -fopenmp -O2

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: FTcom 

FTcom: main.cpp Tucker.o Tensor.o util.o Option.o
	$(CXX) $(CXXFLAGS) -o FTcom main.cpp Tucker.o Tensor.o util.o Option.o $(LIB_FLAGS)

Tucker.o: util.h Tensor.h Tucker.h Tucker.cpp 
	$(CXX) $(CXXFLAGS) -c -o Tucker.o Tucker.cpp $(LIB_FLAGS) 

Tensor.o: util.h Tensor.h Tensor.cpp
	$(CXX) $(CXXFLAGS) -c -o Tensor.o Tensor.cpp $(LIB_FLAGS) 

util.o: util.h Tensor.h util.cpp
	$(CXX) $(CXXFLAGS) -c -o util.o util.cpp $(LIB_FLAGS) 

Option.o: Option.h Option.cpp
	$(CXX) $(CXXFLAGS) -c -o Option.o Option.cpp $(LIB_FLAGS) 

.PHONY: clean

clean:
	rm -f FTcom *.o

