import cv2
from ultralytics import YOLO
import click
from utils import draw_yolo as draw
from loguru import logger

@click.command()
@click.argument('model_path', default='model/yolov8n.pt')
@click.argument('video_path', default='/home/SENSETIME/yangchengwei/coding/jupyter-notebook/data/cars.mp4')
@click.option('--output_path', '-o', default='./out/cars_result.avi')
def run(model_path, video_path, output_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    logger.debug(f"fps: {fps}, size: {size}, source: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    # create window to display image really time
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", size[0], size[1])
    for i in range(600):
        ret, frame = cap.read()
        if ret:
            img0 = frame
            img = img0
            results = model.predict(source=img, project="detect", name="crowd")
            draw(img0, 1, 1, results[0])
            cv2.imshow("result", img0)
            if cv2.waitKey(40) == ord('q'):
                break
            out.write(img0)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    run()
