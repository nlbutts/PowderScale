import numpy as np
import cv2
#from pytesseract import image_to_string
import pytesseract
import time
import glob
import re
import tensorflow as tf

def resize(img, max_width, max_height):
    w = img.shape[1]
    h = img.shape[0]
    
    if w > max_width:
        ratio = max_width / w
        new_height = int(h * ratio)
        img = cv2.resize(img, (max_width, new_height)) 
    
    w = img.shape[1]
    h = img.shape[0]
    if h > max_height:
        ratio = max_height / h
        new_width = int(w * ratio)
        img = cv2.resize(img, (new_width, max_height)) 

    return img

def display(img, title = 'img', width = 1000, height = 600, delay = 1):
    img = resize(img, width, height)            
    cv2.imshow(title, img)
    cv2.waitKey(delay)

def display_channels(img):
    cv2.imshow("Red", img[:,:,2])
    cv2.imshow("Green", img[:,:,1])
    cv2.imshow("Blue", img[:,:,0])
    cv2.waitKey(1)

def generate_grid(imgs, grid_width, grid_height):
    new_img = np.zeros((grid_height, grid_width, 3), np.uint8)

    num_imgs = len(imgs)
    # always a grid of 2
    img_width = grid_width // 2
    img_height = int(grid_height / 3)
    
    x = 0
    y = 0
    for i in imgs:
        ri = resize(i, img_width, img_height)
        if len(ri.shape) == 2:
            # Convert to color
            ri = cv2.cvtColor(ri, cv2.COLOR_GRAY2BGR)
        new_img[y:y + ri.shape[0], x:x + ri.shape[1], :] = ri

        x += img_width
        if x >= grid_width:
            x = 0
            y += img_height

    return new_img
    
def get_first_last_nonzero(array, threshold = 0.9):
    nonzeros = np.nonzero(array > threshold)
    start = nonzeros[0][0]
    stop  = nonzeros[0][-1]
    return start, stop
    

def crop_blue(img, debug=False):
    b = img[:,:,0]
    bb = cv2.blur(b, (15,15))
    ret, t = cv2.threshold(bb, 200, 255, cv2.THRESH_BINARY)
    image, contours, heirarchy = cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

    display_rect = cv2.boundingRect(sorted_contours[0])
    cropped = img[display_rect[1]:display_rect[1] + display_rect[3], display_rect[0]:display_rect[0] + display_rect[2], :]

    rotated = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if debug:
        display(rotated , "Cropped")
   
    rotated = rotated[:, 50::]
    
    return rotated 

def take_picture(should_save=False, d_id=0):
  cam = cv2.VideoCapture(d_id)
  s, img = cam.read()
  if s:
    if should_save:
      cv2.imwrite('ocr.jpg',img)
    print("picture taken")
  return img

def plot_contours(img, high = 4000, low = 500, max_arc = 250, delay = 1):
    image, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)
    draw_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    cx_array = []
    cy_array = []
    
    for i in range(len(contours)):
        cnt = cv2.approxPolyDP(top_cntrs[i], 1, True)
        area = cv2.contourArea(cnt)
        arc = cv2.arcLength(cnt, True)
        
        if (area > low) and (area < high) and (arc < max_arc):
            temp = np.random.rand(3) * 255            
            color = (int(temp[0]), int(temp[1]), int(temp[2]))
            draw_img = cv2.drawContours(draw_img, [cnt], 0, color, -1)
            display(draw_img, 'Contours', delay = delay)

            mom = cv2.moments(cnt)
            cx = mom['m10'] / mom['m00']
            cy = mom['m01'] / mom['m00']
            cx_array.append(cx)
            cy_array.append(cy)
            print('Contour: {:} -- area: {:} -- arc: {:} -- cx: {:}, cy: {:}'.format(i, area, arc, cx, cy))
            time.sleep(delay)


def process_contours(img, high = 4000, low = 500, max_arc = 250, offset = 30, sep_threshold = 50, debug = False):
    image, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)
    draw_img = np.zeros(img.shape, np.uint8)
    cx_array = []
    cy_array = []
    for i in range(len(contours)):
        cnt = cv2.approxPolyDP(top_cntrs[i], 1, True)
        area = cv2.contourArea(cnt)
        arc = cv2.arcLength(cnt, True)
        
        if (area > low) and (area < high) and (arc < max_arc):
            draw_img = cv2.drawContours(draw_img, [cnt], 0, 255, -1)

            mom = cv2.moments(cnt)
            cx = mom['m10'] / mom['m00']
            cy = mom['m01'] / mom['m00']
            cx_array.append(cx)
            cy_array.append(cy)


    cx_sorted = sorted(cx_array)
    diffx = np.diff(cx_sorted)
        
    #print(cx_sorted)
    #print(diffx)
    
    digits = []
    start = int(cx_sorted[0] - offset)
    for i, d in enumerate(diffx):
        if d > sep_threshold:
            stop = int(cx_sorted[i] + diffx[i] // 2)
            #print("i: {:} -- start: {:} -- stop: {:}".format(i, start, stop))
            digits.append(draw_img[0:draw_img.shape[0], start:stop])
            start = stop
   
    if debug:
        plot_contours(img, high, low, max_arc, 0.25)
            
    return draw_img, digits

def find_connected_lines(img, threshold = 80, min_length = 30, gap = 10, debug = False):
    img = img.copy()
    hough = cv2.HoughLinesP(img, 1, np.pi/180, threshold = threshold, minLineLength = min_length, maxLineGap = gap)
    disp_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if hough is not None:
        print('Lines found: {:}'.format(hough.shape[0]))
        for i in hough:
            start = (i[0][0], i[0][1])
            stop  = (i[0][2], i[0][3])
            
            img = cv2.line(img, start, stop, 255, 1)
            
            if debug:
                temp = np.random.rand(3) * 255            
                color = (int(temp[0]), int(temp[1]), int(temp[2]))                
                disp_img = cv2.line(disp_img, start, stop, color, 1)
                cv2.imshow('hough', disp_img)


    if debug:
        display(disp_img, 'hough', width=1000)

    return img


def apply_correction(img):
    width = img.shape[1]
    height = img.shape[0]
    p = [  -0.2,  255.56208031]
    l = np.polyval(p, np.arange(width))
    l = 255 - l
    l = np.reshape(l, (1, len(l)))
    ones = np.ones((height, 1), np.uint8)
    correction = ones * l
    x = img + correction
    y = np.maximum(x, 0)
    y = np.minimum(y, 255)
    y = y.astype('uint8')
    return y

def preprocess_image(img, debug = False):
    blur_amt = 7
    morph_kernel = 11
    canny_threshold1 = 20
    canny_threshold2 = 40

    cropped = crop_blue(img)
    #gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cropped[:, :, 1]
    blur = cv2.blur(gray, (blur_amt,blur_amt))
    correction = apply_correction(blur)
    # Trial and error
    edged_image = cv2.Canny(correction, canny_threshold1, canny_threshold2)
    #edged_image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    
    # This kernel produced nice looking edges
    morph_image= cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel , morph_kernel)))
    
    # This produced nice contour images
    #plot_contours(morph_image, 4000, 100, 1)
    filled_image, digits = process_contours(morph_image, 2400, 500, 250)
    
    rect_digits = []
    for d in digits:
        r = cv2.boundingRect(d)
        d = d[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        rect_digits.append(d)

    # This is to connect the characters together
    #filled_image2 = cv2.morphologyEx(filled_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
    
    # These parameters connected gaps in the numbers together
    #hough_img = find_connected_lines(filled_image2, 50, 100, 50, False)
    #display(hough_img)

    if debug:
        imgs = [cropped, blur, correction, edged_image, morph_image, filled_image]        
        debug_img = generate_grid(imgs, 1600, 900)
        display(debug_img, "debug", width = 1600, height=900)
    return rect_digits

def tf_mnist_train():
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    model.save('mninst_nlb.h5')
    
    return model
    
def tf_mninst_load(file):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    
    model.load_weights(file)
    return model
    
def tf_test(model, digits):
    x = np.zeros((len(digits), 28, 28))
    
    for i, d in enumerate(digits):
        small = cv2.resize(d, (28,28))
        x[i, :, :] = small

    y = model.predict(x)
    
    num = 0   
    for v in y:
        index = [i for i, e in enumerate(v) if e > 0]
        num *= 10
        num += index[0]
        
    return num    
   
def create_training_data(cap, take_every_n_frame = 30, train_width = 10, train_height = 40):

    train_data = []
    response_data = []

    img_count = 0    
    ret, img = cap.read()
    while ret:
        ret, img = cap.read()
        if img_count < take_every_n_frame:
            img_count += 1
        else:
            img_count = 0;
            
            digits = preprocess_image(img, True)
            # Even images are training images, odd are test
            print("Found {:} digits".format(len(digits)))
            for i, d in enumerate(digits):
                display(d, 'train')
                print(d.shape)
                key = cv2.waitKey(0)
                cv2.destroyWindow('train')
                print("Key pressed: {:}".format(key))
                if key != ord(' '):
                    response_data.append(key)
                    small = resize(d, train_width, train_height)
                    small = small.reshape((small.shape[0] * small.shape[1]))
                    small = np.concatenate((small, np.zeros(train_width * train_height - small.shape[0])))
                    train_data.append(small)
                    print('Inserted training data')
    
   
    np.save('train_data', train_data)
    np.save('response_data', response_data)
    
def train():
    train_data = np.load('train_data.npy')
    response_data = np.load('response_data.npy')
    
    # Convert the response data into one hot encoding
    y_train = np.zeros((len(response_data), 10), np.float32)
    response_data -= ord('0')
    for i, v in enumerate(response_data):
        y_train[i, v] = 1        

    train_data = train_data.astype('float32')    
    response_data = response_data.astype('float32')

    #data = cv2.ml.TrainData_create(train_data, cv2.ml.ROW_SAMPLE, y_train)
    model = cv2.ml.KNearest_create()
    model.train(train_data, cv2.ml.ROW_SAMPLE, response_data )

    return model
    
def ml_test(model, digits, train_width = 10, train_height = 40):
    for d in digits:
        small = resize(d, train_width, train_height)
        small = small.reshape((small.shape[0] * small.shape[1]))
        small = np.concatenate((small, np.zeros(train_width * train_height - small.shape[0])))
        small = small.astype('float32')
        small = small.reshape((1, small.shape[0]))
        ret, results = model.predict(small)
        print(ret)
        print(results)

def run():
    cap = cv2.VideoCapture('../data/test.mp4')        
    #model = tf_mninst_load('checkpoints/mninst')
    model = train()
    while cap.isOpened():
        ret, img = cap.read()
        while ret:
            ret, img = cap.read()
            digits = preprocess_image(img, True)
            ml_test(model, digits)
            #num = tf_test(model, digits)            
            print('Detected number: {:}'.format(num))
            time.sleep(0.25)

    
#train()    
#tf_mnist_train()
#run()
print('Hello')
