.PHONY: clean build
BUILD_TYPE ?= Debug
CUDA_VERSION := $(shell nvcc --version|grep -oP 'cuda_(\d+\.\d+)'| sed 's/cuda_//')
BUILD_TYPE ?= Debug
target ?= cuda
build: clean
	@echo "CUDA Version: $(CUDA_VERSION)"
	cd ${target} && \
	cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DCUDA_VERSION=${CUDA_VERSION}

	cd ${target}/build && make -j8

clean:
	rm -rf ${target}/build
