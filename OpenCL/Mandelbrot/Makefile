
ifndef CC
	CC = gcc
endif

CCFLAGS= -O3

LIBS = -lglut -lGLU -lGL -lm -lOpenCL -g

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

mandel: mandel_interactive.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f mandel
