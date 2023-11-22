import torch
from ultralytics import YOLO
## 是否有用，待定
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model = YOLO('model/yolov8n.pt')
# net.eval()
# dummpy_input = torch.randn(1, 3, 640, 640)
# torch.onnx.export(net, dummpy_input, 'model/yolov8n.onnx', export_params=True,
#                   input_names=['input'],
#                   output_names=['output'])
success = model.export(format="onnx", simplify=True)
assert success
print("转换 over")
