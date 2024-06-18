Compile=g++ -g -std=c++20 -O3 # improve run time e.g. 10s to 4s
biaxial_polymer: main.o biaxial_polymer.o
	$(Compile) -o biaxial_polymer main.o biaxial_polymer.o
main.o: main.cpp biaxial_polymer.h
	$(Compile) -o main.o -c main.cpp
biaxial_polymer.o: biaxial_polymer.cpp biaxial_polymer.h
	$(Compile) -o biaxial_polymer.o -c biaxial_polymer.cpp
clean:
	rm -f main.o biaxial_polymer.o