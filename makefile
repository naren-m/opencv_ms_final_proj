INCLUDES = `pkg-config --cflags opencv` 
CFLAGS = -g -Wall 
LIBS = `pkg-config --libs opencv` 



obj : objectTracking.o
	g++ $(CFLAGS) -o obj objectTracking.o $(LIBS)

objectTracking.o : objectTracking.cpp
	g++ $(CFLAGS) -c $(INCLUDES) objectTracking.cpp 




clean :
	-rm *.o  obj
