opencv_url="https://github.com/opencv/opencv/archive/refs/tags/4.8.1.zip"
opencv_contrib_url="https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.1.zip"

wget --progress=bar:force $opencv_url -O opencv.zip
wget --progress=bar:force $opencv_contrib_url -O opencv_contrib.zip

unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.8.1

sudo apt-get -y install  build-essential checkinstall cmake pkg-config yasm  libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev  libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils libtiff5-dev libjpeg62-dev ffmpeg libatlas-base-dev gfortran
