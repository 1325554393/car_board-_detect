import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# img_path = "./111.jpg"
# img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
# # cv2.imshow("img",img)
# # cv2.waitKey()
# # b ,g, r = cv2.split(img)
# # img = cv2.merge([r,g,b])
# print(img.shape)

# # #img_arr = np.array(img.convert("L"))
# plt.imshow(img,cmap="gray")
# plt.show()



# # x_test = []
# # x_test.append(img_arr)
# # x_test.append(img_arr)
# # print(x_test[0].shape)
# # x_save = np.reshape(x_test, (len(x_test), -1))
# # print(x_save.shape)
# # np.save("./aaa.npy", x_save)
# 
# 
# 准备标签模板
template = ['0','1','2','3','4','5','6','7','8','9',
			'A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
			'藏','川','鄂','甘','赣','贵','桂','黑','沪','吉','冀','津','晋','京','辽','鲁','蒙','闽','宁',	
			'青','琼','陕','苏','皖','湘','新','渝','豫','粤','云','浙'] 

def gerenate_dataset(img_path):
	file = os.listdir(img_path)
	labels = []
	imgs = []
	for i in file:
		for j in os.listdir(img_path+"/"+i):
			labels.append(template.index(i))
			# cv2读取中文路径时的方法，将文件读取到内存，再读取，0是代表以灰度读取
			img = cv2.imdecode(np.fromfile(img_path+"/"+i+"/"+j,dtype=np.uint8),0)
			img = img/255.
			imgs.append(img)

	# cv2.imshow("6666",imgs[7666])
	# cv2.waitKey()
	imgs = np.array(imgs)
	imgs = np.reshape(imgs, (len(imgs), 20, 20,1))
	labels = np.array(labels, dtype=np.int64)

	return imgs, labels

def preprocessing(x_imgs, y_labels):
	"""相同的随机种子，使图片和标签一一对应"""
	np.random.seed(120)
	np.random.shuffle(x_imgs)
	np.random.seed(120)
	np.random.shuffle(y_labels)

	x_train = x_imgs[:-1000]
	y_train = y_labels[:-1000]

	x_test = x_imgs[-1000:]
	y_test = y_labels[-1000:]

	# print(x_test.shape)

	return x_train, y_train, x_test, y_test

def conv_model(input_shape,num_classes):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32,(1,1), input_shape=input_shape, activation="relu",padding="SAME"))
	model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu",padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

	model.add(tf.keras.layers.Conv2D(32,(3,3), activation="relu", padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D())

	model.add(tf.keras.layers.Conv2D(64,(3,3), activation="relu", padding="SAME"))
	model.add(tf.keras.layers.MaxPooling2D())

	model.add(tf.keras.layers.Flatten())
	# model.add(tf.keras.layers.Dense(512,activation="relu"))
	# model.add(tf.keras.layers.Dropout(0.5))

	model.add(tf.keras.layers.Dense(128,activation="relu"))
	model.add(tf.keras.layers.Dropout(0.5))

	# model.add(tf.keras.layers.Dense(256,activation="relu"))
	# model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(num_classes,activation="softmax"))

	return model

def zdy(x_train, y_train, x_test, y_test):
	"""自定义卷积神经网络训练"""
	# batch配对
	train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
	test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

	# 假设只有两层的卷积
	w1 = conv_w((1,1,1,32))
	b1 = conv_b(32)

	w2 = conv_w((3,3,32,32))
	b2 = conv_b(32)

	# 一层全连接层
	w3 = conv_w((20*20*32, 65))
	b3 = conv_b(65)

	# 训练
	epochs = 10
	lr_base = 0.2
	lr_decay = 0.99

	for epochs in range(epochs):
		for step, (x_train, y_train) in enumerate(train_db):
			lr = lr_base*lr_decay**epochs
			with tf.GradientTape() as tape:
				y1 = tf.matmul(x_train, w1) + b1
				y2 = tf.matmul(y1 , w2) + b2
				y2_reshape = tf.reshape(y2, [-1, 20*20*32])
				y3 = tf.matmul(y2_reshape, w3) + b3
				y3 = tf.nn.softmax(y3)
				y_ = tf.one_hot(y_train, depth=65)
				# 均方误差求损失
				loss = tf.reduce_mean(tf.square(y_ - y))
			# 实现梯度更新
			grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

			w1.assign_sub(lr * grads[0])
			b1.assign_sub(lr * grads[1])
			w2.assign_sub(lr * grads[2])
			b2.assign_sub(lr * grads[3])
			w3.assign_sub(lr * grads[4])
			b3.assign_sub(lr * grads[5])

	pass

def conv_w(input_shape):
	return tf.Variable(tf.random.truncated_normal(input_shape), stddev=0.1, seed=1)

def conv_b(input_shape):
	return tf.Variable(tf.random.truncated_normal(input_shape), stddev=0.1, seed=1)

def draw(history):
	acc = history.history["sparse_categorical_accuracy"]
	val_acc = history.history["val_sparse_categorical_accuracy"]
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]

	plt.subplot(1,2,1)
	plt.plot(acc, label="Training Accuracy")
	plt.plot(val_acc, label="Validation Accuracy")
	plt.title("Training and Validation Accuracy")

	plt.subplot(1,2,2)
	plt.plot(loss, label="Training Loss")
	plt.plot(val_loss, label="Validation Loss")
	plt.title("Training and Validation Loss")

	plt.legend()
	plt.show()


def main():
	input_shape = (20,20,1)
	num_classes = 65
	img_path = "./template"
	save_imgs_path = "./imgs_dataset.npy"
	save_labels_path = "./labels_dataset.npy"

	if os.path.exists(save_imgs_path) and os.path.exists(save_labels_path):
		print("load_datasets-----------------")
		x_imgs = np.load(save_imgs_path)
		x_imgs = np.reshape(x_imgs, (len(x_imgs), 20, 20,1))
		# cv2.imshow("6666",x_imgs[7666])
		# cv2.waitKey()
		y_labels = np.load(save_labels_path)
	else:
		print("gerenate_dataset-----------------")
		x_imgs, y_labels = gerenate_dataset(img_path)

		print("save_dataset-----------------")
		x_imgs = np.reshape(x_imgs, (len(x_imgs), -1))
		np.save(save_imgs_path, x_imgs)
		np.save(save_labels_path, y_labels)

	x_train, y_train, x_test, y_test = preprocessing(x_imgs, y_labels)

	model = conv_model(input_shape,num_classes)

	checkpoint_save_path = "./checkpoint/carboard.ckpt"
	
	# model.summary()
	# 模型编译 ，优化器，损失函数
	model.compile(optimizer="Adam",
					loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
					metrics=["sparse_categorical_accuracy"])
	# 断点续训
	
	if os.path.exists(checkpoint_save_path+".index"):
		print("load_model_weight------------")
		model.load_weights(checkpoint_save_path)

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
													save_weights_only=True,
													save_best_only=True)
	# 模型训练
	history = model.fit(x_train, y_train, batch_size=32, epochs=5, 
				validation_data=(x_test, y_test), validation_freq=1,
				callbacks=[cp_callback])

	draw(history)
	

	"""
	测试
	-----------------------------------------------------
	img_pre = "./111.jpg"
	img_pre =  cv2.imread(img_pre, cv2.IMREAD_GRAYSCALE)
	img_pre = img_pre/255.0
	# print(img_pre.shape)
	img_pre = np.reshape(img_pre, (20,20,1))
	img_pre = img_pre[tf.newaxis, ...]

	model.load_weights(checkpoint_save_path)
	reslut = model.predict(img_pre)
	pred = tf.argmax(reslut, axis=1)
	print(template[int(pred.numpy())])
	"""

if __name__ == '__main__':
	main()