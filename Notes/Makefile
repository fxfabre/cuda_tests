# Makefile

NVCC = g++
CFLAGS = 
LDFLAGS = 
PROG = prog

OBJECTS = main.o

%.o : %.cpp
	$(NVCC) -c %^ $(CFLAGS) -o $@
	$(NVCC) -M %^ $(CFLAGS) > $@.dep
	
%.o : %.c
	$(NVCC) -c %^ $(CFLAGS) -o $@
	$(NVCC) -M %^ $(CFLAGS) > $@.dep

%.o : %.cu
	$(NVCC) -c %^ $(CFLAGS) -o $@
	$(NVCC) -M %^ $(CFLAGS) > $@.dep

include $(wildcard *.dep) /dev/null

all : $(OBJECTS)
	$(NVCC) $(OBJECTS) $(LDFLAGS) -o $(PROG)

clean :
	$(RM) $(OBJECTS) *.dep

