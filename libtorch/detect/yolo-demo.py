
import cv2
import  matplotlib.pyplot as plt
from ultralytics import YOLO
import click
import os
from utils import draw_yolo as draw


@click.command()
@click.argument('model_path', default='model/yolov8n.pt')
@click.argument('img_path', default='data/crowd.jpeg')
def run(model_path, img_path):
    img0 = cv2.imread(img_path)
    # 直接跑YOLO 并不需要前处理
    img = img0
  

    model = YOLO(model_path)
    results = model.predict(source=img, project="detect", name="crowd")

  
    draw(img0, 1, 1, results[0])
    plt.imshow(img0)
    plt.show()
    if not os.path.exists('out'):
        os.makedirs('out')
    cv2.imwrite('out/yolo-output.jpg', img)

if __name__ == '__main__':
    run()

