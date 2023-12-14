
set -x
set +e
sudo apt-get -y install libgtk2.0-dev


set -e
opencv_url="https://github.com/opencv/opencv/archive/refs/tags/4.8.1.zip"
opencv_contrib_url="https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.1.zip"

cwd=$(pwd)
tgt=$1/opencv_install
if [ -d "$tgt" ]; then
    echo "removing $tgt"
    rm -rf $tgt;
fi
mkdir -p $tgt
cd $tgt
wget --progress=bar:force $opencv_url -O opencv.zip
wget --progress=bar:force $opencv_contrib_url -O opencv_contrib.zip

unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.8.1
ln -s $cwd/Makefile .
make build
