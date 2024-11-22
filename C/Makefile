FLAGS=-O3 -fopenmp

CC=g++

RM=rm -f

EXEC=RiemannSiegel

all: $(EXEC)

$(EXEC): $(EXEC).cpp
	$(CC) $(FLAGS) $(EXEC).cpp -o $(EXEC)

clean:
	$(RM) $(EXEC)
