import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import tensorflow as tf

emotions = ["Angry", "Happy", "Neutral", "Surprise"]
# emotions = ["Angry", "Happy"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = RandomForestClassifier(min_samples_leaf=20)
data = {}


def get_files(emotion):
    print("./save/%s/*" % emotion)
    files = glob.glob("./save/%s/*" % emotion)
    print(len(files))

    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]

    return training, prediction


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(0, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        landmarks_vectorised = []
        real = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            # meannp = np.asarray((ymean,xmean))
            # coornp = np.asarray((z,w))
            # dist = np.linalg.norm(coornp-meannp)
            # anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi)
            # landmarks_vectorised.append(dist)
            # landmarks_vectorised.append(anglerelative)
        for i in range(0, 8):
            landmarks_vectorised.append(0)
        for i in range(0,16):
            # print(real)
            # print(len(real))
            real.extend(landmarks_vectorised)
        data['landmarks_vectorised'] = real

    if len(detections) < 1:
        landmarks_vectorised_x = "error"


def make_sets():
    training_labels = []
    training_data = []
    prediction_labels = []
    prediction_data = []
    # Angry = [1, 0]
    # Happy = [0, 1]
    Angry = [1, 0, 0, 0]
    Happy = [0, 1, 0, 0]
    Neutral = [0, 0, 1, 0]
    Surprise = [0, 0, 0, 1]
    length = 0  # count of prediction file
    for emotion in emotions:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face training")
            else:
                training_data.append(data['landmarks_vectorised'])
                if emotions.index(emotion) == 0:
                    training_labels.append(Angry)
                elif emotions.index(emotion) == 1:
                    training_labels.append(Happy)
                elif emotions.index(emotion) == 2:
                    training_labels.append(Neutral)
                else:
                    training_labels.append(Surprise)
                # if emotions.index(emotion) == 0:
                #     training_labels.append(Angry)
                # elif emotions.index(emotion) == 1:
                #     training_labels.append(Happy)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            #	    cv2.imwrite('pred_im_%i.jpg' %length, clahe_image)
            length += 1
            if data['landmarks_vectorised'] == "error":
                print("no face prediction")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    #		npar_pred = np.array(prediction_data)
    #		pred_pro = clf.predict_proba(npar_pred)
    return training_data, training_labels, prediction_data, prediction_labels


training_data, training_labels, prediction_data, prediction_labels = make_sets()
# training_data.append(0)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 2304])  # variable to use input layer
y_ = tf.placeholder(tf.float32, [None, 2])  # variable to use output layer
W = tf.Variable(tf.zeros([2304, 2]))  # 144*10 matrix
b = tf.Variable(tf.zeros([2]))  # 10 list
y = tf.nn.softmax(tf.matmul(x, W) + b)  # x*w+b


# make matrix size and return
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# make 0.1 and return wanna size
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # add padding to don't reduce matrix
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# layer1 - 12*12 matrix make 32ro using max pool
x_image = tf.reshape(x, [-1, 48, 48, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# layer2 - input is 14*14 matrix 32ro, output matrix 7*7 matrix 64ro
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# layer3 - last data is 7*7*64 = 3136 but using 1024
W_fc1 = weight_variable([12 * 12 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer - 1024 node make percentage 10ro(0~9) using soft max
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train & save model
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    # training_data,training_labels, prediction_data, prediction_labels
    if i % 100 == 0:
        print(len(training_data[0]))
        print(training_data[0])
        # print(len(training_data))
        print(len(training_labels))
        print(training_labels)
        train_accuracy = accuracy.eval(feed_dict={x: training_data, y_: training_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # batch[0] 28*28 image, [1] is number tag, keep_prob : dropout percentage
    train_step.run(feed_dict={x: training_data, y_: training_labels, keep_prob: 0.5})
save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: training_data, y_: training_labels, keep_prob: 1.0}))
sess.close()

# accur_lin = []
"""for i in range(0,1):
    print("Making sets %s" %i)
    training_data, training_labels, prediction_data, prediction_labels = make_sets()

    npar_train = np.array(training_data)
    npar_trainlabs = np.array(training_labels)
    
    print("training model %s" %i)
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" %i)
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("Model: ", pred_lin)
    accur_lin.append(pred_lin)
    pred_pro = clf.predict_proba(npar_pred)
    for row in range(0,30):
        for col in range(0,3):
	    temp_maxpred = np.float(pred_pro[row][col]).tolist()
	maxpred = max(temp_maxpred)
        maxpred_list = list.index(maxpred)
	print maxpred_list
	if maxpred_list == 0:
	    feature ="Disgust"
	elif maxpred_list == 1:
	    feature = "Angry"
	elif maxpred_list == 2:
	    feature = "Fear"
	else:
	    feature = "Surprise"
	print(feature)
print("Mean value accuracy in Model: %.3f" %np.mean(accur_lin)) 
"""
#    cv2.imwrite("pred_im_%s.jpg" %feature,

# pred_pro = clf.predict_proba(npar_pred)
# for i in range(0,length):
#    image = cv2.imread("pred_im_%i.jpg" %i)
#    cv2.imshow("predicted file", image)
#    cv2.waitKey(0)
# print("expected : %i %i" %i pred_pro)
