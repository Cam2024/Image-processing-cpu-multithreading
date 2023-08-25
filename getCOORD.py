from PIL import Image
import csv

def find_black_and_white_pixels(image_path, x_min, x_max, y_min, y_max):
    image = Image.open(image_path)
    width, height = image.size

    # 确保输入的坐标范围在图像的尺寸内
    x_min = max(x_min, 0)
    x_max = min(x_max, width)
    y_min = max(y_min, 0)
    y_max = min(y_max, height)

    black_white_pixels = []

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            pixel = image.getpixel((x, y))
            if pixel == (0, 0, 0) or pixel == (255, 255, 255):
                black_white_pixels.append((x, y))

    return black_white_pixels

def save_to_csv(coordinates, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y"])
        for coord in coordinates:
            writer.writerow([coord[0], coord[1]])

# 输入图像路径和坐标范围
image_path = "frame_020.png"
x_min, x_max = 370, 1550
y_min, y_max = 200, 880

# 查找符合条件的像素坐标
coordinates = find_black_and_white_pixels(image_path, x_min, x_max, y_min, y_max)

# 保存坐标到CSV文件
output_file = "new.csv"
save_to_csv(coordinates, output_file)

print("已将符合条件的像素坐标保存到new.csv文件中。")
