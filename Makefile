
all: lib

prog: dlml/pearsonr.c
	gcc -Wall -g -O3 -c dlml/pearsonr.c
	gcc pearsonr.o -lgsl -lgslcblas -lm

lib: dlml/pearsonr.c
	gcc -fPIC -shared -o libcorr.so dlml/pearsonr.c -lgsl -lgslcblas -lm

clean:
	rm -f libcorr.so a.out pearsonr.o
