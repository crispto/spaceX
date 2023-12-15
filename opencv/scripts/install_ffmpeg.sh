#see https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu
sudo apt-get update -qq && sudo apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libmp3lame-dev \
  libsdl2-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev

VERSION=$(cat /etc/os-release | awk -F= '$1=="VERSION_ID"{print $2}' | sed 's/"//g')

if [ $VERSION == "20.04"];then
    sudo apt install -y libunistring-dev libaom-dev libdav1d-dev
fi

