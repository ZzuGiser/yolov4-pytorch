#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from road_yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    # img = input('Input image filename:')
    img = "./road_images/1_2682_3769.jpg"
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
