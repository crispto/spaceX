.PHONY: clean build
BUILD_TYPE ?= Debug
CUDA_VERSION := $(shell nvcc --version|grep -oP 'cuda_(\d+\.\d+)'| sed 's/cuda_//')
CUDA_VERSION ?= 11.4
BUILD_TYPE ?= Debug

repo ?= opencv
target ?= geem

build: clean
	@echo "CUDA Version: $(CUDA_VERSION)"
	cd ${repo} && \
	cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DCUDA_VERSION=${CUDA_VERSION}

	cd ${repo}/build && make -j8

geem:
	cmake --build cuda/build --target geem
clean:
	rm -rf ${repo}/build

run:
	@${repo}/build/bin/${target} 1
	
