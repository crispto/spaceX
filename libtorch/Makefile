run-yolo-demo:
	python detect/yolo-demo.py ./model/yolov8n.pt ./data/crowd.jpeg

run-onnx-demo-gpu:
	python detect/onnx_demo.py ./model/yolov8n.onnx ./data/crowd.jpeg --use-gpu true

run-video-demo:
	python ./detect/video_demo.py model/yolov8n.pt /home/SENSETIME/yangchengwei/data/yizhuang-172/38.2023-06-08_16-07-17.mp4 -o ./out/test.avi

onnx-bench:
	python ./detect/onnx_bench.py model/yolov8n.onnx /home/SENSETIME/yangchengwei/900G/coding/jupyter-notebook/data/cars.mp4 -o out -s False -e cpu -b -1

onnx2trt:
	trtexec --onnx=model/yolov8n.onnx --saveEngine=model/yolov8n.trt --fp16 --workspace=4096 
