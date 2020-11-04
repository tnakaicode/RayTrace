#compile = nvcc -arch=sm_20
compile = nvcc -arch=sm_20 -DOUBLE
g.exe:  rhs.cu evolve.cu gode.cu rhs.h gode.h
	$(compile) -dc -g -G rhs.cu
	$(compile) -dc -g -G evolve.cu 
	$(compile)  -c -g -G gode.cu
	$(compile) -g -G rhs.o evolve.o gode.o -o g.exe
clean:
	rm -f *.o *.exe
