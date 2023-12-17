set -e
set -x
root=$HOME/Desktop/tmp
if [ ! -d "$root" ]; then
    mkdir -p $root
fi
cd $root
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && sudo make install


sudo apt-get install  -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
cd $root
git clone https://git.ffmpeg.org/ffmpeg.git

cd ffmpeg/

./configure --enable-nonfree \
--enable-cuda-nvcc \
--enable-cuda --enable-nvenc --enable-cuvid \
--enable-libnpp \
--extra-cflags=-I/usr/local/cuda/include \
--extra-ldflags=-L/usr/local/cuda/lib64 \
--disable-static \
--enable-shared

make -j 8

sudo make install


# /usr/local/bin/ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i /home/data2/video.mp4 -c:a copy -c:v h264_nvenc output.mp4


