cmd="./build/opencv_demo"
# run cmd
eval "$cmd 200"

pprof $cmd ./profil2.prof --text
