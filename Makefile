# ============================================================
#                 CONFIGURATION DU PROJET
# ============================================================

CUDA_HOME ?= /usr/local/cuda

NVCC := $(CUDA_HOME)/bin/nvcc

GO := go

TARGET := xor-cuda

CUDA_LIBRARY := libcuda_nn.so


# ============================================================
#                 ARCHITECTURE GPU
# ============================================================

# RTX A6000 = Ampere = sm_86
#
# Pour une autre carte :
#
#   make CUDA_ARCH=sm_89
#
# par exemple pour une architecture Ada compatible.

CUDA_ARCH ?= sm_86


# ============================================================
#                 OPTIONS CUDA
# ============================================================

NVCC_FLAGS := \
	-O3 \
	-arch=$(CUDA_ARCH) \
	-Xcompiler -fPIC


# ============================================================
#                         CIBLES
# ============================================================

.PHONY: all
all: $(TARGET)


# ============================================================
#                    BIBLIOTHÈQUE CUDA
# ============================================================

$(CUDA_LIBRARY): cuda_nn.cu cuda_nn.h

	$(NVCC) \
		$(NVCC_FLAGS) \
		-shared \
		cuda_nn.cu \
		-o $(CUDA_LIBRARY)


# ============================================================
#                       PROGRAMME GO
# ============================================================

$(TARGET): main.go $(CUDA_LIBRARY)

	CGO_ENABLED=1 \
	CGO_CFLAGS="-I." \
	CGO_LDFLAGS="-L. -lcuda_nn -L$(CUDA_HOME)/lib64 -lcudart" \
	$(GO) build \
		-o $(TARGET) \
		.


# ============================================================
#                          RUN
# ============================================================

.PHONY: run
run: $(TARGET)

	LD_LIBRARY_PATH=.:$(CUDA_HOME)/lib64 \
		./$(TARGET)


# ============================================================
#                         CLEAN
# ============================================================

.PHONY: clean
clean:

	rm -f $(TARGET)
	rm -f $(CUDA_LIBRARY)
