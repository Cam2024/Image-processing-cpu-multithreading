

import cv2
import csv

def change_coordinates_to_white(input_image_path, csv_file_path, output_image_path):
    # 读取图片
    image = cv2.imread(input_image_path)

    # 读取CSV文件中的坐标数据
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过CSV文件的标题行
        for row in csv_reader:
            x, y = map(int, row)
            # 将坐标对应的像素点颜色设置为白色 (255, 255, 255)
            image[y, x] = [255, 255, 255]

    # 保存修改后的图片
    cv2.imwrite(output_image_path, image)

# 使用示例
input_image_path = "frame_020.png"  # 输入图片的路径
csv_file_path = "new.csv"        # CSV文件的路径
output_image_path = "age8.jpg" # 输出图片的路径

change_coordinates_to_white(input_image_path, csv_file_path, output_image_path)
