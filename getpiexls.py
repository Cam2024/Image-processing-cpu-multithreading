import pandas as pd
from PIL import Image

# 读取CSV文件
csv_file_path = "ccc.csv"
data = pd.read_csv(csv_file_path, header=None, names=["y", "x"])

# 获取图像尺寸
image_path = "frame_020.png"
image = Image.open(image_path)
width, height = image.size

# 获取像素的RGB信息
pixels_info = []
for index, row in data.iterrows():
    y, x = row["y"], row["x"]
    if 0 <= x < width and 0 <= y < height:
        pixel_rgb = image.getpixel((x, y))
        pixels_info.append((y, x, pixel_rgb[0], pixel_rgb[1], pixel_rgb[2]))

# 将像素信息保存到新的CSV文件中
pixels_df = pd.DataFrame(pixels_info, columns=["y", "x", "R", "G", "B"])
pixels_csv_file_path = "pixels.csv"
pixels_df.to_csv(pixels_csv_file_path, index=False)

print(f"已保存像素信息到 {pixels_csv_file_path}")
