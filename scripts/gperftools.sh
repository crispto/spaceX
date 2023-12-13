# 安装 gperftools
function install(){
    sudo apt -y install libunwind8-dev
    git clone git@github.com:gperftools/gperftools.git
    cd gperftools
    sh autogen.sh
    ./configure
    make all
    sudo make install
}

install
