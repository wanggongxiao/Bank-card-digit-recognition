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
cv_show("template image",template)
gray_template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
cv_show("gray_template",gray_template)

# 做二值化处理
thresh_template = cv2.threshold(gray_template,100,255,cv2.THRESH_BINARY_INV)[1]
cv_show("thresh_template",thresh_template)

# 检测轮廓
cnts,hierarchy = cv2.findContours(thresh_template.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template,cnts,-1,(0,0,255),3)
cv_show("contours",template)

# 将检测到的框从左到右排序
refcnts = utils.sort_contours(cnts)[0]

digits = {}


