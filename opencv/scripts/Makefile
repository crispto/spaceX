.PHONY: build
build:
	cd build && \
	cmake \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.8.1/modules \
	-D CMAKE_BUILD_TYPE=Debug \
	-D CMAKE_INSTALL_PREFIX=./install \
	-D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUBLAS=ON \
	-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
	-D CUDA_ARCH_BIN=6.1 \
	-D CUDA_ARCH_PTX="" \
	-D WITH_NVCUVID=ON \
	-D CUDA_GENERATION=Auto \
	-D BUILD_TESTS=ON \
	-D BUILD_PERF_TESTS=ON \
	-D BUILD_EXAMPLES=ON \
	-D BUILD_opencv_apps=ON \
	-D ENABLE_PROFILING=ON ..

	make -j$(shell nproc)

