INCLUDES = `pkg-config --cflags opencv` 
CFLAGS = -g -Wall 
LIBS = `pkg-config --libs opencv` 



cvt : convertHsv32.o
	g++ $(CFLAGS) -o cvt convertHsv32.o $(LIBS)

convertHsv32.o : convertHsv32.cpp
	g++ $(CFLAGS) -c $(INCLUDES) convertHsv32.cpp 


clean :
	-rm *.o  cvt 
