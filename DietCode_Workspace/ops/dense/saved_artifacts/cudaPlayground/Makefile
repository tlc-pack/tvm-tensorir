include ../../../shared/CUDA.mk

.PHONY: all clean

all: build

KERNEL_EXES := main.exe

build: $(KERNEL_EXES)

%.o: %.cu Makefile
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

%.exe: %.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

.PHONY: clean
clean:
	$(RM) $(KERNEL_EXES)
