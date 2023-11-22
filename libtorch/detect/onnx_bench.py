import onnx
import onnxruntime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import click
from utils import nice_colors
from loguru import logger
import os 
import psutil
import time
from utils import draw_onnx as draw
import cProfile, pstats, io
from pstats import SortKey


def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
        box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
        box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter

# onnx 输出的是 bbox 和 80 个类别的置信度，所以需要进行 NMS
from math import exp
def soft_nms(pred, conf_thres, sigma=0.4):
    """使用 soft nms 进行 NMS

    Args:
        pred (_type_): Onnx predict output [8400, 85], [x1, y1, x2, y2, max_conf, *all_conf]
        conf_thres (_type_): 小于该阈值的框被过滤
        sigma: 衰减因子
    """
    pred = pred[np.where(pred[:, 4] > conf_thres)]
    conf_all = pred[:, 5:]
    label_result = np.argmax(conf_all, axis=-1)
    cls_all = list(set(label_result))
    pred = np.insert(pred, 5, label_result, axis=-1) # 将 label 插入到第5
    pred = pred[..., :7] # 后面的分数抛弃
    # pred 格式为 [center_x, center_y, w, h, max_conf, label, max_conf_for_iter]
    pred[:, 6] = pred[:, 4]
    output_box = []
    for cls in cls_all:
      cls_box = pred[np.where(pred[:, 5] == cls)]
      
      # 按得分逆序排列
      sort_indics = np.argsort(-cls_box[:, 4])
      cls_box = cls_box[sort_indics]
      # cls_box.sort(key=-1*cls_box[4])
      while len(cls_box):
        M = cls_box[0]
        output_box.append(M)
        cls_box = np.delete(cls_box, 0, axis=0)
        to_delete = []
        for i, remain in enumerate(cls_box):
          iou = getIou(M, remain, getInter(M, remain))
          remain[6] *= exp(-1 * iou ** 2 / sigma)
          # logger.info(f"remain[4] is {remain[4]}")
          if remain[6] < conf_thres:
            to_delete.append(i)
        cls_box = np.delete(cls_box, to_delete, axis=0)
    return output_box   


def nms(pred, conf_thres, iou_thres):
    conf = pred[..., 4] > conf_thres  # 只处理最大置信度大于阈值的框
    box = pred[conf == True]
    cls_conf = box[..., 5:]  # 这些都是列别分数
    cls = []
    # 定位每个最大值的 index
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))

    # 对每一种类别进行 NMS
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []  # 候选框
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])

        cls_box = np.array(cls_box)
        # sort by conf
        cls_box = cls_box[np.argsort(-cls_box[..., 4])]
        # box_conf = cls_box[..., 4]
        # box_conf_sort = np.argsort(box_conf)
        # max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        max_conf_box = cls_box[0]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)

        while len(cls_box) > 0:
            # 刚刚插入的最大框
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def get_onnx_class_map(model_path: str):
  """加载 Onnx 模型并通过 onxx api 获取 label 到 name 的映射

  Args:
      model_path (str): 模型路径

  Returns:
      dict: label -> name
  """
  import ast
  model = onnx.load(model_path)
  for meta in model.metadata_props:
      if meta.key == 'names':
          classmap = ast.literal_eval(meta.value)
          return classmap
import time 

@click.command()
@click.argument('model_path', default='model/yolov8n.onnx')
@click.argument('video_path', default='data/crowd.jpeg')
@click.option('--outpath', '-o', type =str, default='out', help='输出路径')
@click.option('--use-soft', '-s', type=bool, default=False, help="是否使用 soft-nms 策略")
@click.option('--executer', '-e', type=click.Choice(['cpu', 'gpu', 'trt']), default='cpu', help="推理模式")
@click.option('--bench-num', '-b', type=int, default=100, help="跑多少帧")
def run(model_path, video_path, outpath, use_soft, executer, bench_num):
    pr = cProfile.Profile()
    pr.enable()
    cap = cv2.VideoCapture(video_path)
    ok, test_img = cap.read()
    if not ok:
      logger.error(f"read video {video_path} failed")
      return
    # 读取视频的宽高
    height0, width0 = test_img.shape[:2]
    logger.info(f"video size: {width0}x{height0}")
    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("result", width0, height0)
    height, width = 640, 640
    x_scale = width0 / width
    y_scale = height0 / height
    
    writer = cv2.VideoWriter(os.path.join(outpath,"detect_result.avi"), cv2.VideoWriter_fourcc(*'MJPG'), 30, (width0, height0))
    label2name = get_onnx_class_map(model_path)
    # Define ONNX Runtime session options
    print(f"provider {onnxruntime.get_available_providers()}")
    assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
    if executer == 'trt':
      sess_options = onnxruntime.SessionOptions()
      # Please change the value according to best setting in Performance Test Tool result.
      sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

      session = onnxruntime.InferenceSession(model_path, sess_options, providers=["TensorrtExecutionProvider"])
      
    if executer=='gpu':
      sess_options = onnxruntime.SessionOptions()
      # Please change the value according to best setting in Performance Test Tool result.
      sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

      session = onnxruntime.InferenceSession(model_path, sess_options, providers=["CUDAExecutionProvider"])
    else:
      sess_options = onnxruntime.SessionOptions()
      sess_options.intra_op_num_threads=psutil.cpu_count(logical=True)

      session = onnxruntime.InferenceSession(model_path, sess_options,  providers=["CPUExecutionProvider"])
      
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    frame_id = 0
    while True:
      ok, img0 = cap.read()
      if not ok:
        break
      # if you set bench_num, then only run bench_num frames, may be useful for benchmark
      if bench_num >0 and frame_id > bench_num:
        break
      frame_id += 1
      if frame_id % 50 == 0:
        logger.debug(f"frame_id: {frame_id}")
      
      img = img0 / 255.
      img = cv2.resize(img, (width, height))
      img = np.transpose(img, (2, 0, 1))
      data = np.expand_dims(img, axis=0)
      # get meta
      
      pred = session.run([output_name], {input_name: data.astype(np.float32)})[0]
      pred = np.squeeze(pred)  # 去掉 batch 维度
      pred = np.transpose(pred, (1, 0))

      # onnx 输出的是 bbox 和 80 个类别的置信度 [8400, 84]
      pred_class = pred[..., 4:]
      pred_conf = np.max(pred_class, axis=-1)
      # 将类别概率插入到原来的 pred 中，方便后面进行 NMS
      pred = np.insert(pred, 4, pred_conf, axis=-1)
      if use_soft:
        result = soft_nms(pred, 0.3, 0.4)
      else:
        result = nms(pred, 0.3, 0.45)  # 进行 NMS
      draw(img0, x_scale, y_scale, result, label2name)
      
      # format
      writer.write(img0)
      # cv2.imshow("result", img0)
      # if cv2.waitKey(40) == ord('q'):
      #     break
    cv2.destroyAllWindows()
    cap.release()
    logger.info(f"run over, executor: {executer}, frame_num: {bench_num}")
    pr.disable()
    
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(10)
    logger.warning("-----------------------------------------------")
    ps.print_stats()
    logger.warning("-----------------------------------------------")
    print(s.getvalue())
    
if __name__ == '__main__':
  
  run()
 
    