.PHONY: clean build

target ?= tensorrt
build: clean
	cd ${target} && \
	cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  && \
	cmake --build build

clean:
	rm -rf ${target}/build
