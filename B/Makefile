FLAGS=-O3 -Wall -std=c++17 -fopenmp -fopenmp-targets=nvptx64
EXEC=TSPSolver
CXX=g++

all: $(EXEC)

$(EXEC): $(EXEC).cpp
	$(CXX) $(FLAGS) $(EXEC).cpp -c -o $(EXEC).o
	$(CXX) $(FLAGS) $(EXEC).o -o $(EXEC)
