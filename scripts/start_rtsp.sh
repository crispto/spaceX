#!/bin/bash
# check if ffmpeg is installed
if [ -z "$(command -v ffmpeg)" ]; then
  echo "Error: ffmpeg is not installed." >&2
  exit 1
fi


function Usage(){
    echo "--- 启动一个rtsp服务"
    echo "Usage: $0 <data/a.mp4>"
}

sourcePath="../data/sample_720p.mp4"
if [ -x "$s1" ]; then
    sourcePath=$1
fi
#  启动一个rtsp服务
image_name="bluenviron/mediamtx:latest"
#check if container already running
if [ -z "$(docker ps |grep $image_name)" ]; then
    docker run --rm -it --network=host $image_name
fi

rtsp_url="rtsp://127.0.0.1:8554/stream"
ffmpeg -re -stream_loop -1 -i $sourcePath -c copy -f rtsp $rtsp_url

echo "rtsp服务启动成功, $rtsp_url"
