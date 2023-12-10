#!/usr/bin/zsh
# 安装
function install(){

git clone git@github.com:brendangregg/FlameGraph.git
cd FlameGraph
sudo update-alternatives --install /usr/bin/stackcollapse-perf.pl stackcollapse `pwd`/stackcollapse-perf.pl 60
sudo update-alternatives --install /usr/bin/flamegraph.pl framegraph `pwd`/flamegraph.pl 60
}

# 测试使用
function test(){

perf record -F 99 -a -g -- sleep 10
sudo chmod -R 777 perf.data
perf script -i perf.data > out.perf
stackcollapse-perf.pl out.perf > out.folded
flamegraph.pl out.folded > kernel.svg
grep cpuid out.folded| flamegraph.pl > cpuid.svg

}

function remove(){
    rm -rf *.data *.perf *.svg *.folded
}

function run(){
    perf record -F99 -a -g deepstream-app -c configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt

}

function frame(){
    perfdata=$1
    if [ ! -f $perfdata ];then
        echo "perf.data not exist"
        exit 1
    fi
    sudo chmod -R 777 $perfdata
    perf script -i $perfdata > out.perf
    stackcollapse-perf.pl out.perf > out.folded
    flamegraph.pl out.folded > kernel.svg
    
}
# 从参数中获取 test 还是 install
case $1 in
    "install")
        install
        ;;
    "test")
        test
        ;;
    "clean")
        remove
        ;;
    "run")
        run $@
        ;;
    "frame")
        frame $2
        ;;
    *)
        echo "Usage: $0 {install|test}"
        exit 2
esac
