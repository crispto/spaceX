.PHONY: clean build

CUDA_VERSION := $(shell nvcc --version|grep -oP 'cuda_(\d+\.\d+)'| sed 's/cuda_//')
target ?= cuda
build: clean
	@echo "CUDA Version: $(CUDA_VERSION)"
	cd ${target} && \
	cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCUDA_VERSION=${CUDA_VERSION}

	cd ${target}/build && make -j8

clean:
	rm -rf ${target}/build
