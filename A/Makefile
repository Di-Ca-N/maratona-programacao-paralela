FLAGS=-O3 -Wall -std=c++17 -mcmodel=large -fopenmp

CC=g++

RM=rm -f

EXEC=optimize

all: $(EXEC)

$(EXEC): $(EXEC).cpp
	$(CC) $(FLAGS) $(EXEC).cpp -o $(EXEC)

clean:
	$(RM) $(EXEC)
