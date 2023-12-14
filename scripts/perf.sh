#!/usr/bin/zsh
# 安装

# help function
function usage(){
    echo "Usage: $0 [install|test|clean|run|frame]"

}

# install frame graph tools, and install perf if not installed
function install(){
    #check if perf is installed
    if ! command -v perf &> /dev/null
    then
        echo "perf could not be found"
        sudo apt-get install linux-tools-common linux-tools-generic linux-tools-`uname -r`
    fi
    git clone git@github.com:brendangregg/FlameGraph.git
    cd FlameGraph
    sudo update-alternatives --install /usr/bin/stackcollapse-perf.pl stackcollapse `pwd`/stackcollapse-perf.pl 60
    sudo update-alternatives --install /usr/bin/flamegraph.pl framegraph `pwd`/flamegraph.pl 60
    cd .. && rm -rf FlameGraph

    #run rest
    if ! perf_test ; then
        echo "perf test failed"
        exit 1
    fi
}

# test perf command on sleep 10
function perf_test(){
    perf record -F 99 -a -g -- sleep 10
    # perf record --call-graph dwarf -a sleep 2
    sudo chmod -R 777 perf.data
    perf script -i perf.data > out.perf
    stackcollapse-perf.pl out.perf > out.folded
    flamegraph.pl out.folded > kernel.svg
    grep cpuid out.folded| flamegraph.pl > cpuid.svg
}

# clear this dir
function clear(){

    set +e
    # clear this dir, if not exist, dont care
    rm -rf *.perf
    rm -rf *.data
    rm -rf  *.svg
    rm -rf *.data.old
    rm -rf *.folded
}


function run(){
    perf record -F99 -a -g $@
    sudo chmod -R 777 perf.data
    perf report -i perf.data
    perf script -i perf.data > out.perf
    stackcollapse-perf.pl out.perf > out.folded
    flamegraph.pl out.folded > kernel.svg
    grep cpuid out.folded| flamegraph.pl > cpuid.svg

}



# 从参数中获取 test 还是 install
while [ "$1" != "" ]; do
    case $1 in
        --help|-h)
            usage
            exit 0
            ;;
        install)
            install
            ;;
        test)
            perf_test
            ;;
        clean)
            clear &>/dev/null
            ;;
        run)
            shift;
            run $@
            exit 0
            ;;
        *)
            usage
            exit 2
    esac
    shift
done
