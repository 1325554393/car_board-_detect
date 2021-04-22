import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import numpy as np
from utils import conv_model

# 准备模板
template = ['0','1','2','3','4','5','6','7','8','9',
			'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
			'藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',	
			'青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙']
			

def cv_show(name, img):
	"""显示图片"""
	cv2.imshow(name, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def plt_show(img):
	"""显示原始图像"""
	b,g,r = cv2.split(img)
	img = cv2.merge([r,g,b])
	plt.imshow(img)
	plt.show()

def plt_show_gray(img_gray):
	plt.imshow(img_gray, cmap="gray")
	plt.show()

def gray_gauss(img):
	"""图像去噪灰度处理"""
	img = cv2.GaussianBlur(img, (3,3), 0) # 高斯去噪，提高边缘检测准确度
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img_gray

def read_directory(directory_name):
	"""读取给定文件夹下所有文件，并返回文件列表"""
	refer_list = []
	for file in os.listdir(directory_name):
		refer_list.append(directory_name+"/"+file)
	return refer_list

def get_chinese_list():
	"""匹配中文"""
	chinese_list = []
	for i in range(34,64): # 对应模板中的下标
		chinese_word = read_directory("./template/"+template[i])
		chinese_list.append(chinese_word)
	return chinese_list

def get_english_list():
	"""匹配英文"""
	english_list = []
	for i in range(0,34):
		english_word = read_directory("./template/"+template[i])
		english_list.append(english_word)
	return english_list

def get_num_list():
	"""匹配后面的字符"""
	num_list = []
	for i in range(0,34):
		num_word = read_directory("./template/"+template[i])
		num_list.append(num_word)
	return num_list

def template_score(template, img):
	"""读取模板地址与图片进行匹配，返回得分"""
	template_img = cv2.imdecode(np.fromfile(template, dtype=np.uint8), 1)
	template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
	ret, template_img = cv2.threshold(template_img, 0, 255, cv2.THRESH_OTSU)
	img_copy = img.copy()
	h, w = img_copy.shape
	template_img = cv2.resize(template_img, (w, h))
	# TM_CCOEFF是相关性系数匹配将模版对其均值的相对值与图像对其均值的相关值进行匹配,
	# 1表示完美匹配,-1表示糟糕的匹配,0表示没有任何相关性(随机序列)
	result = cv2.matchTemplate(img_copy, template_img, cv2.TM_CCOEFF)
	return result[0][0] # 取出具体数值

"""
模板匹配精度不高
修改成卷积网络分类识别
def template_matching(word_images,chinese_list, english_list, num_list):
	results = []
	for index, word_img in enumerate(word_images):
		if index == 0:
			best_score = []
			for chinese_words in chinese_list:
				score = []
				for chinese_word in chinese_words:
					result = template_score(chinese_word, word_img)
					score.append(result)
				best_score.append(max(score))
			i = best_score.index(max(best_score))
			r = template[34+i]
			results.append(r)
		if index == 1:
			best_score = []
			for english_words in english_list:
				score = []
				for english_word in english_words:
					result = template_score(english_word,word_img)
					score.append(result)
				best_score.append(max(score))
			i = best_score.index(max(best_score))
			r = template[10+i]
			results.append(r)
		else:
			best_score = []
			for num_words in num_list:
				score = []
				for num_word in num_words:
					result = template_score(num_word, word_img)
					score.append(result)
				best_score.append(max(score))
			i = best_score.index(max(best_score))
			r = results.append(r)
	return results
"""

def get_result(model,word_images):
	checkpoint_save_path = "./checkpoint/carboard.ckpt"
	if os.path.exists(checkpoint_save_path+".index"):
		# print("load_model_weight------------")
		model.load_weights(checkpoint_save_path)
	else:
		print("train the model first-------")

	res = []
	for img in word_images:
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if img[i][j] > 80:
					img[i][j] = 0
				else:
					img[i][j] = 255
		# cv2.imshow("1", img)
		# cv2.waitKey()
		img = img/255.0
		# 尺寸的变化导致识别精度低
		img = np.resize(img, (20,20,1))
		img = img[tf.newaxis, ...]
		result = model.predict(img)
		pred = tf.argmax(result, axis=1)
		res.append(template[int(pred.numpy())])
		# print(template[int(pred.numpy())])
	return res

def get_carboard_position(img):
	"""得到图片中车牌的位置"""
	img_copy = img.copy()
	img_gray = gray_gauss(img)
	# sobel算子边缘检测（做一个y方向的检测）
	# 此处可通过调整sobel算子的参数，提高边缘检测准度
	Sobel_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0)
	Sobel_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1)
	abs_x = cv2.convertScaleAbs(Sobel_x) # 转回unit8格式
	abs_y = cv2.convertScaleAbs(Sobel_y)
	# 图像混合加权
	# https://blog.csdn.net/zh_jessica/article/details/77992578?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161500680016780271517078%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161500680016780271517078&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-77992578.first_rank_v2_pc_rank_v29&utm_term=cv2.addweighted
	dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0) 
	# plt_show_gray(dst)
	# plt_show_gray(abs_y)
	# 自适应阈值处理,二值化
	# https://blog.csdn.net/weixin_51025273/article/details/113594576?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161500598816780262578535%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161500598816780262578535&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-113594576.first_rank_v2_pc_rank_v29&utm_term=cv2.threshold
	ret, img = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
	# ret, img = cv2.threshold(abs_x, 0, 255, cv2.THRESH_OTSU)
	# plt_show_gray(img)
	# 闭运算，是白色部分练成整体,膨胀：白色部分变大，17>5,在X方向膨胀力度更大
	# cv2.getStructuringElement( ) 返回指定形状和尺寸的结构元素
	kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT,(17,5))
	# print(kernel_x)
	# https://blog.csdn.net/qq_39507748/article/details/104539673?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161500872216780271569071%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161500872216780271569071&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-104539673.first_rank_v2_pc_rank_v29&utm_term=cv2.morphologyEx
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel_x,iterations=1)
	# plt_show_gray(img)
	# 去除一些小的白点
	kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT,(20,1))
	kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT,(1,19))
	# 膨胀，腐蚀
	img = cv2.dilate(img, kernel_x)
	img = cv2.erode(img, kernel_x)

	img = cv2.erode(img, kernel_y)
	img = cv2.dilate(img, kernel_y)

	# plt_show_gray(img)

	# 中值滤波去除噪点
	img = cv2.medianBlur(img, 5)
	# plt_show_gray(img)

	# 轮廓的检测
	# cv2.RETR_EXTERNAL表示只检测外轮廓
	# cv2.CHAIN_APPROX_SIMPLE压缩水平方向， 垂直方向， 对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只保留4个点的信息
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 绘制轮廓
	# cv2.drawContours(img_copy, contours, -1, (0,0,255), 3)
	# plt_show(img_copy)
	# 筛选出车牌位置的轮廓
	# 这里只做一个车牌长宽在3:1到4:1之间的一个判断
	# print(contours) 点坐标
	for item in contours:
		rect = cv2.boundingRect(item)
		# print(rect) # 矩阵坐标
		x = rect[0]
		y = rect[1]
		w = rect[2]
		h = rect[3]
		if (w>(h*3)) and (w<(h*5)):
			img = img_copy[y:y+h, x:x+w]
			# plt_show(img)
			return img

def carboard_splite(img):
	img_gray = gray_gauss(img)
	img_copy = img_gray
	ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
	# plt_show_gray(img_gray)
	# 计算二值图像黑白点的个数，处理绿牌照的问题，让车牌号码始终为白色
	area_white = 0
	area_black = 0
	h, w = img_gray.shape
	for i in range(h):
		for j in range(w):
			if img_gray[i,j] == 255:
				area_white += 1
			else:
				area_black += 1
	if area_white > area_black:
		# 统计白色区域的个数多于黑色，白色区域为背景
		# 颜色反转
		ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
		# plt_show_gray(img_gray)
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 卷积核，用于形态学后处理
		img_gray = cv2.dilate(img_gray, kernel)
		img_gray = cv2.erode(img_gray, kernel)
		# plt_show_gray(img_gray)
		# 绘制轮廓
		contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# print(contours)
		# cv2.drawContours(img, contours, -1, (0,0,255), 2)
		# plt_show(img)
		words = []
		word_images = []
		for item in contours:
			# print(item)
			word = []
			rect = cv2.boundingRect(item) # boundingRect用最小的矩阵，把找到的形状包起来
			x = rect[0]
			y = rect[1]
			w = rect[2]
			h = rect[3]
			# 匿名函数车牌的排序，x方向的升序
			word.append(x)
			word.append(y)
			word.append(w)
			word.append(h)
			words.append(word)
			words = sorted(words,key=lambda s:s[0],reverse=False)
			# 判断合格的矩形框
			if h>1.8*w and h<3*w and y>10 and y<59:
				img_split = img_copy[y:y+h, x:x+w]
				# plt_show_gray(img_split)
				word_images.append(img_split) 
		return word_images

def main():
	# 卷积网络输入参数
	input_shape = (20,20,1)
	num_classes = 65
	# 识别图片信息
	img = cv2.imread("./test.jpg")
	# plt_show(img)
	# img_gray = gray_gauss(img)
	# plt_show_gray(img_gray)
	# cv_show("img", img)
	img_carboard = get_carboard_position(img)

	# os.makedir(path)
	i = 0
	word_images = carboard_splite(img_carboard)
	if os.path.exists("./recognization"):
		pass
	else:
		for img in word_images:
			i += 1
			cv2.imwrite(("./recognization/word_%d.png" % i), img)

	# word_images = np.resize(word_images, (len(word_images), 20, 20, 1))
	# print(word_images[1])

	# cv2.imshow("1",word_images[1])
	# cv2.waitKey()

	"""
	for i ,j in enumerate(word_images):
		plt.subplot(1,8,i+1)
		plt.imshow(word_images[i], cmap="gray")
		# plt.imshow(word_images[i])
	plt.show()
	"""

	

	# chinese_list = get_chinese_list()
	# english_list = get_english_list()
	# num_list = get_num_list()

	# word_imgs_copy = word_images.copy
	# results = template_matching(word_images,chinese_list, english_list, num_list)
	# print(results)

	model = conv_model(input_shape,num_classes)
	res = get_result(model, word_images)
	height, weight = img.shape[0:2]
	cv2.rectangle(img, (int(0.2*weight), int(0.75*height)), (int(weight*0.8), int(height*0.95)), (0, 255, 0), 3)
	cv2.putText(img, "".join(res[0:8]), (int(0.2*weight)+1, int(0.75*height)+50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
	plt_show(img)
	

if __name__ == '__main__':
	main()