import cv2
import numpy as np
import argparse


import utils.utils as utils

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-t","--template",required=True,help="Path to the template image")
args = vars(ap.parse_args())

# 图像绘制
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# print(args["template"])
template = cv2.imread(args["template"])
# cv_show("template image",template)
gray_template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
# cv_show("gray_template",gray_template)

# 做二值化处理
thresh_template = cv2.threshold(gray_template,100,255,cv2.THRESH_BINARY_INV)[1]
# cv_show("thresh_template",thresh_template)

# 检测轮廓
cnts,hierarchy = cv2.findContours(thresh_template.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
temp = template.copy()
cv2.drawContours(temp,cnts,-1,(0,0,255),3)
# cv_show("contours",temp)

# 将检测到的框从左到右排序
refcnts = utils.sort_contours(cnts)[0]
cv2.drawContours(template,refcnts,-1,(0,255,0),2)
# cv_show("temp",template)
digits = {}

# 遍历每一个轮廓
for (i,c) in enumerate(refcnts):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = gray_template[y:y + h,x:x + w]
    roi = cv2.resize(roi,(57,88))
    # cv_show("roi", roi)
    # print(f"数字 {i} 的轮廓位置和大小：x={x}, y={y}, w={w}, h={h}")
    # 每一个模板代表一个数字
    digits[i] = roi

# for i, roi in digits.items():
#     cv_show("roi",roi)

# 初始化卷积核
rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
Sqkernel   = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 读取输入图像，预处理
image = cv2.imread(args["image"])
# cv_show("image",image)

image = utils.resize(image,width=300)
# 做灰度图处理
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show("grey",grey)
# thresh_image = cv2.threshold(grey,180,255,cv2.THRESH_BINARY_INV)[1]
# cv_show("thresh_image",thresh_image)
tophat = cv2.morphologyEx(grey,cv2.MORPH_TOPHAT,rectkernel)
# cv_show("tophat",tophat)

gradx = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradx = np.absolute(gradx)
(minVal, maxVal) = (np.min(gradx), np.max(gradx))
gradx = (255 * ((gradx - minVal) / (maxVal - minVal)))
gradx = gradx.astype("uint8")
cv_show("gradx",gradx)

# 闭操作：先膨胀再腐蚀
gradx = cv2.morphologyEx(gradx,cv2.MORPH_CLOSE,rectkernel)
cv_show("gradx",gradx)

thresh = cv2.threshold(gradx, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show('thresh',thresh)


# 找到边界框
ctns,res = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cur_image = image.copy()
cv2.drawContours(cur_image,ctns, -1,(0,0,255),3)
print(cnts.__sizeof__())
cv_show("gradx",cur_image)

locs = []

# 遍历轮廓
for(i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)

    # 根据比例筛选出需要的数据
    if ar >2.5 and ar < 5.0:
        if (w > 40 and w < 50) and (h >10 and h <20):
            locs.append((x,y,w,h))

locs = sorted(locs,key=lambda x:x[0])
output = []

for(i,(gx,gy,gw,gh)) in enumerate(locs):
    groudOutput = []
    group = grey[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]