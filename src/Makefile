OBJS = util.o random.o reduce.o density.o 

NVCC = /usr/local/cuda/bin/nvcc
CUDA_HOME = /usr/local/cuda
CUDA_INCLUDE = $(CUDA_HOME)/include
CUDA_LIB = /usr/local/cuda/lib64

R_HOME = /usr/lib/R
R_INCLUDE = /usr/share/R/include
R_LIB = 

CC = gcc -std=gnu99
CFLAGS = -g -O2 -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -g
CPPFLAGS = 
CPICFLAGS = -fpic

RPATH = -Xlinker -rpath="/usr/local/cuda/lib64"
LDFLAGS = 
LFLAGS=-lcurand -lcufft  ## curand

#compiler/preprocessor options
#INCLUDES = -I. -I"$(CUDA_INCLUDE)" -I"$(R_INCLUDE)"
INCLUDES = -I"$(CUDA_INCLUDE)" -I"$(R_INCLUDE)"
LIBS = -L"$(CUDA_LIB)" -lcublas $(R_LIB)

TARGETS = gpda.so

all: $(TARGETS)

$(TARGETS): $(OBJS)
	$(NVCC) -arch=sm_35 -shared $(RPATH) $(LDFLAGS) $(LIBS) $(LFLAGS) $(OBJS) -o $@ 
 
$(OBJS): %.o: %.cu
	$(NVCC) -arch=sm_35 -c -Xcompiler "$(CPICFLAGS) $(CPPFLAGS) $(CFLAGS)" $(INCLUDES) $^ -o $@ 

clean:
	rm -rf *o

.PHONY: all clean
