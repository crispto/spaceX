- export_to_onnx: 将 yolov8 模型转换为 onnx 格式
- onnx_demo  使用 onnx 进行 yolov8 的模型推理
- yolo-demo 直接使用 YOLO 模型跑图片检测推理和可视化
- utils 里包含可视化代码
- video_demo 使用 onnx 进行视频流的检测推理和可视化
- yolo-demo 直接使用 YOLO 模型跑图片检测推理和可视化

BemcharkResult

|| Tensorrt | CUDA  | CPU
|---| --- | --- | ---|
|Time/sec| 0.053 | 0.012 | 0.052

```
#Tensorrt
  ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     3599  191.341    0.053  191.378    0.053 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:202(run)
     3599   68.933    0.019   68.933    0.019 {method 'write' of 'cv2.VideoWriter' objects}
        2    0.000    0.000   44.343   22.172 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:358(__init__)
        2   44.343   22.172   44.343   22.172 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:434(_create_inference_session)
     3599   28.002    0.008   28.002    0.008 {resize}
     3601   11.062    0.003   11.062    0.003 {method 'read' of 'cv2.VideoCapture' objects}
     3599    1.002    0.000    7.621    0.002 900G/coding/github/laboratory/libtest/libtorch/./scripts/detect/onnx_bench.py:84(nms)
   209675    4.462    0.000    5.173    0.000 900G/coding/github/laboratory/libtest/libtorch/./scripts/detect/onnx_bench.py:25(getInter)
     4245    3.182    0.001    3.182    0.001 {method 'astype' of 'numpy.ndarray' objects}
     3599    0.371    0.000    1.682    0.000 900G/coding/github/laboratory/libtest/libtorch/scripts/detect/utils.py:13(draw_onnx)


## GPU
      ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     3599   53.246    0.015   53.246    0.015 {method 'write' of 'cv2.VideoWriter' objects}
     3599   43.188    0.012   43.226    0.012 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:202(run)
     3599    1.165    0.000   11.728    0.003 900G/coding/github/laboratory/libtest/libtorch/./detect/onnx_bench.py:84(nms)
     3599    9.605    0.003    9.605    0.003 {resize}
   209673    7.682    0.000    8.640    0.000 900G/coding/github/laboratory/libtest/libtorch/./detect/onnx_bench.py:25(getInter)
     3601    6.349    0.002    6.349    0.002 {method 'read' of 'cv2.VideoCapture' objects}
        1    0.000    0.000    4.579    4.579 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:358(__init__)
        1    4.578    4.578    4.579    4.579 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:434(_create_inference_session)
     4245    3.347    0.001    3.347    0.001 {method 'astype' of 'numpy.ndarray' objects}
     3599    0.543    0.000    2.000    0.001 900G/coding/github/laboratory/libtest/libtorch/detect/utils.py:13(draw_onnx)

#CPU
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     3599  188.420    0.052  188.455    0.052 miniconda3/envs/dl/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:202(run)
     3599   70.219    0.020   70.219    0.020 {method 'write' of 'cv2.VideoWriter' objects}
     3599   27.517    0.008   27.517    0.008 {resize}
     3601   11.191    0.003   11.191    0.003 {method 'read' of 'cv2.VideoCapture' objects}
     3599    1.017    0.000    7.851    0.002 900G/coding/github/laboratory/libtest/libtorch/./detect/onnx_bench.py:84(nms)
   209675    4.587    0.000    5.338    0.000 900G/coding/github/laboratory/libtest/libtorch/./detect/onnx_bench.py:25(getInter)
     4245    3.207    0.001    3.207    0.001 {method 'astype' of 'numpy.ndarray' objects}
     3599    0.385    0.000    1.712    0.000 900G/coding/github/laboratory/libtest/libtorch/detect/utils.py:13(draw_onnx)
     3599    0.811    0.000    0.949    0.000 miniconda3/envs/dl/lib/python3.10/site-packages/numpy/lib/function_base.py:5368(insert)
    20253    0.943    0.000    0.943    0.000 {putText}

```
