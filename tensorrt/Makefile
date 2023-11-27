
CUDA_VERSION := $(shell nvcc --version|grep -oP 'cuda_(\d+\.\d+)'| sed 's/cuda_//')

ifeq ($(CUDA_VERSION), 11.4)
	model := ../libtorch/model/yolov8n_cuda11_4.engine
else
	model := ../libtorch/model/yolov8n_cuda12_0.engine
endif

.PHONY: build
run:
	./build/sampleDetect  ../libtorch/data/crowd.jpeg -e $(model)


dump-engine:
	./build/sampleDetect -m /home/SENSETIME/yangchengwei/900G/coding/github/spaceX/libtorch/model/yolov8n.onnx

build:
	cmake -DCMAKE_BUILD_TYPE=Debug \
	-B build .
	make -C build -j8
