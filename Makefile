CC=gcc
CFLAGS=-Wall -Wextra -g
OFLAGS=-Ofast -fopenmp -funroll-loops -finline-functions
MFLAGS=-lm

all: init copy reduc dotprod triad

init: init.c
	$(CC) $(CFLAGS) $(OFLAGS) $< ../../../ynotif/ynotif.c ../../../ydata/ydata.c -o $@ $(MFLAGS)

copy: copy.c
	$(CC) $(CFLAGS) $(OFLAGS) $< ../../../ynotif/ynotif.c ../../../ydata/ydata.c -o $@ $(MFLAGS)

reduc: reduc.c
	$(CC) $(CFLAGS) $(OFLAGS) $< ../../../ynotif/ynotif.c ../../../ydata/ydata.c -o $@ $(MFLAGS)

dotprod: dotprod.c
	$(CC) $(CFLAGS) $(OFLAGS) $< ../../../ynotif/ynotif.c ../../../ydata/ydata.c -o $@ $(MFLAGS)

triad: triad.c
	$(CC) $(CFLAGS) $(OFLAGS) $< ../../../ynotif/ynotif.c ../../../ydata/ydata.c -o $@ $(MFLAGS)

clean:
	rm -Rf init copy reduc dotprod triad
