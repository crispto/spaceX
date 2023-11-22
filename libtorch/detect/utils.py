import colorsys
import numpy as np
import cv2
def generate_nice_colors(num_colors):
    # Generate a list of distinct colors
    hsv_colors = [(i / num_colors, 1.0, 1.0) for i in range(num_colors)]
    rgb_colors = [tuple(int(value * 255)
                        for value in colorsys.hsv_to_rgb(*hsv)) for hsv in hsv_colors]
    return rgb_colors
# Generate 80 nice colors
nice_colors = generate_nice_colors(80)

def draw_onnx(img, xscale, yscale, bboxes, label2name):
  """可视化ONNX 模型 .onnx （经过NMS）的结果

  Args:
      img (_type_): _description_
      xscale (_type_): _description_
      yscale (_type_): _description_
      bboxes (_type_): [center_x, center_y, width, height, score, label]
      label2name (dict): int -> label name 
  """
  for box in bboxes:
      pos = box[:4]
      # onnx 输出的是中心点坐标和宽高，需要转换成左上角和右下角坐标
      pos = [pos[0] - pos[2]/2, pos[1] - pos[3] /
              2, pos[0] + pos[2]/2, pos[1] + pos[3]/2]
      score = box[4]
      label = int(box[5])
      name = label2name[label]
      # print(f"{name}: {score}, {pos}, {pos2}, {label}")
      text = f"{name}: {score:.2f}"
      # draw rectrange and label with cv2
      font_size = 0.7
      lefttop = (int(pos[0]*xscale), int(pos[1]*yscale))
      rightbottom = (int(pos[2]*xscale), int(pos[3]*yscale))
      cv2.rectangle(img, lefttop, rightbottom, (0, 255, 0), 2)
      cv2.rectangle(img, lefttop, (lefttop[0]+len(text)*12,
                                    lefttop[1] - 14), nice_colors[len(nice_colors)-label-1], -1)
      cv2.putText(img, text, lefttop, cv2.FONT_HERSHEY_SIMPLEX,
                  font_size, (255, 255, 255), 2)

def draw_yolo(img, xscale, yscale, result, label2name=None):
  """可视化 yolo .pth 模型推理的结果

  Args:
      img (_type_): _description_
      xscale (_type_): _description_
      yscale (_type_): _description_
      result (_type_): yolo Results结构体,参见 https://docs.ultralytics.com/modes/predict/#working-with-results
      label2name (_type_, optional): _description_. Defaults to None.
  """
  label2name = result.names
  for box in result.boxes:
    pos = box.xyxy.cpu().numpy().squeeze().astype(np.int16).tolist()
    score = box.conf.cpu().item()
    label = int(box.cls.cpu().item())
    name = label2name[label]
    # print(f"{name}: {score}, {pos}, {pos2}, {label}")
    text = f"{name}: {score:.2f}"
    # draw rectrange and label with cv2
    font_size = 0.7
    lefttop = (int(pos[0]*xscale), int(pos[1]*yscale))
    rightbottom = (int(pos[2]*xscale), int(pos[3]*yscale))
    cv2.rectangle(img, lefttop, rightbottom, (0, 255, 0), 2)
    cv2.rectangle(img, lefttop, (lefttop[0]+len(text)*12, lefttop[1]- 14), nice_colors[len(nice_colors)-label-1], -1)
    cv2.putText(img, text, lefttop, cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), 2)
    # draw rectrange and label with plt

