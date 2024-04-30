import os
import json
from PIL import Image, ImageDraw, ImageFont
import cv2

# 设置海报模板路径、文字内容、输出目录等参数
template_path = 'boy_autumn.png'  # 海报模板路径
text_lines = [
    {'content': '悠闲的夏日', 'position': (300, 150), 'color': (0, 0, 0, 0)},
    {'content': ' 悠闲的夏日\n不在乎目的地，在乎的是沿途的风景,以及看风景的心情。', 'position': (300, 200), 'color': (0, 255, 0)},
    {'content': '以及看风景的心情。', 'position': (300, 250), 'color': (0, 0, 255)}
]  # 文字内容、位置和颜色的列表

text_lines = [
    {'content': '悠闲的夏日，', 'position': (300, 150), 'color': (255, 255, 255, 0)},  # 淡青色
    {'content': '不在乎目的地，', 'position': (310, 210), 'color': (126, 200, 190,1)},  # 淡绿色
    {'content': '在乎的是沿途的风景', 'position': (330, 270), 'color': (16, 175, 220,1)},  # 淡绿色
    {'content': '以及看风景的心情。', 'position': (360, 330), 'color': (13, 180, 180,1)}  # 淡黄色
]   # 文字内容、位置和颜色的列表
text_lines = auto_text
output_dir = 'output_posters'  # 输出目录
font_path = '/Library/Fonts/Alibaba-PuHuiTi-Heavy.otf'  # 字体文件路径
font_size = 42  # 字体大小
text_angle = 290  # 文字旋转角度（竖排）

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载模板图片
template_image = Image.open(template_path)

# 创建一个可以在Pillow中使用的字体对象
font = ImageFont.truetype(font_path, font_size)

# 创建一个可以在Pillow中使用的绘图对象
draw = ImageDraw.Draw(template_image)

# 在指定位置添加文字
for line in text_lines:
    draw.text(line['position'], line['content'], font=font, fill=line['color'], rotation=text_angle)

# 保存处理后的图片
output_path = os.path.join(output_dir, 'poster_with_multiple_lines.jpg')
template_image.save(output_path)

# 如果需要进行图层融合和模糊处理，可以使用OpenCV
# 读取处理后的图片
image = cv2.imread(output_path)

# 这里可以添加OpenCV的图层融合和模糊处理代码
# 例如，使用高斯模糊
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 保存模糊处理后的图片
cv2.imwrite(os.path.join(output_dir, 'blurred_poster.jpg'), blurred_image)

print("海报生成和处理完成。")
