FLAGS=-O3 -fopenmp -pg

CC=gcc

RM=rm -f

EXEC=center

all: $(EXEC)

$(EXEC): $(EXEC).c
	$(CC) $(FLAGS) $(EXEC).c -o $(EXEC)

clean:
	$(RM) $(EXEC)
