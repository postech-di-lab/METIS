CXX = g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS = -g -fopenmp -fPIC -pipe -O3

all: train-ocam build-gcsc

train-ocam: train-ocam.cpp macros.h ocam.h util.o csc-idx.o ocam.o 
	${CXX} ${CXXFLAGS} -o train-ocam train-ocam.cpp ocam.o util.o csc-idx.o

build-gcsc: build-gcsc.cpp macros.h ocam.h util.o csc-idx.o ocam.o
	${CXX} ${CXXFLAGS} -o build-gcsc build-gcsc.cpp ocam.o util.o csc-idx.o

ocam.o: ocam.cpp util.o csc-idx.o
	${CXX} ${CXXFLAGS} -c -o ocam.o ocam.cpp

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp

csc-idx.o: csc-idx.h csc-idx.cpp util.o
	${CXX} ${CXXFLAGS} -c -o csc-idx.o csc-idx.cpp

clean:
	rm -rf  train-ocam build-gcsc *.o 

