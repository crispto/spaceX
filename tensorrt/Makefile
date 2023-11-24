.PHONY: build
run:
	./build/sampleDetect ../libtorch/data/crowd.jpeg -e /home/SENSETIME/yangchengwei/900G/coding/github/spaceX/libtorch/model/yolov8n.engine


dump-engine:
	./build/sampleDetect -m /home/SENSETIME/yangchengwei/900G/coding/github/spaceX/libtorch/model/yolov8n.onnx

build:
	cmake -DCMAKE_BUILD_TYPE=Debug \
	-B build .
	make -C build -j8
