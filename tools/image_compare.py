# 统计两个图像的相似度
import numpy as np
import click
import cv2
from rich.console import Console

console = Console()
def mse(imageA, imageB):
	# 计算两张图片的MSE相似度
	# 注意：两张图片必须具有相同的维度，因为是基于图像中的对应像素操作的
    # 对应像素相减并将结果累加起来
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	# 进行误差归一化
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# 返回结果，该值越小越好，越小说明两张图像越相似
	return err


@click.command()
@click.argument('pa', type=click.Path(exists=True))
@click.argument('pb', type=click.Path(exists=True))
def run(pa, pb):
  a = cv2.imread(pa)
  b = cv2.imread(pb)
  
  mse_value = mse(a, b)
  if mse_value < 10:
    console.print(f"mse: {mse_value}", style="bold green")
  else:
    console.print(f"mse: {mse_value}", style="bold red")
 
if __name__ == "__main__":
  run()
