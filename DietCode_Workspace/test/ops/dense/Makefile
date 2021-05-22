include ../shared/CUDA.mk

# Target rules
.PHONY: all clean

KERNEL_SOURCES := $(wildcard *.cu)
KERNEL_SASSES := $(KERNEL_SOURCES:.cu=.S)

all: $(KERNEL_SASSES)

%.S: %.o Makefile 
	cuobjdump --dump-sass $< > $@

%.o: %.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $< --ptxas-options=-v

clean:
	$(RM) *.S
