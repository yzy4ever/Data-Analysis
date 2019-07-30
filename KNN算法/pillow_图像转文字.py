from PIL import Image

fh=open("C:/应用程序/MyPrograms/Python/KNN算法/QRcodeResult.txt","a")
im=Image.open("C:/应用程序/MyPrograms/Python/KNN算法/QRcode.jpg")
#im.save(".bmp")   #保存为bmp格式
width=im.size[0]  #图片的宽
height=im.size[1]   #图片的高
k=im.getpixel((1,9))   #获取(1,9)的rgb

for i in range(0,width):
    for j in range(0,height):
        cl=im.getpixel((j,i))
        if((cl[0]+cl[1]+cl[2])<20):   #如果rgb为000即黑色  <20 灰色宽容度
            fh.write('1')
        else:
            fh.write('0')
    fh.write('\n')
fh.close()

