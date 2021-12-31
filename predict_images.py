from PIL import Image
from utils.utils_deeplab import DeepLabV3

image_path = 'Image_samples/sheep.jpg'  # 测试图片路径

deeplab = DeepLabV3()

image = Image.open(image_path)
image = deeplab.segmentate_image(image)
image.save('result.jpg')
image.show()