opencv_url="https://github.com/opencv/opencv/archive/refs/tags/4.8.1.zip"
opencv_contrib_url="https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.1.zip"

wget --progress=bar:force $opencv_url -O opencv.zip
wget --progress=bar:force $opencv_contrib_url -O opencv_contrib.zip

unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.8.1
if [ -d "build" ]; then
  rm -rf build
fi
mkdir build && cd build
#get cores of cpu

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


make -j$(nproc)
