from gradio_client import Client
 #建立后台服务器链接
client = Client("http://0.0.0.0:6006")
#查看请求参数
client.view_api(return_format="dict")

#传参请求生成图
out_data = client.predict("An adorable girl with curly hair, innocently laughing with a big smile, looking very happy Poster style --ar 9:16","dpm-solver",14,4.5,0,True)
#生成图片可视化
import matplotlib.pyplot as plt
from PIL import Image

# 图片地址
image_path = out_data[0]

# 打开并显示图片
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')  # 关闭坐标轴
plt.show()
