import numpy as np
import cv2
import time
import glob
import re

def resize(img, max_width, max_height):
    """Image resize helper. This reszie function will intelligently resize
    the image to maintain the aspect ratio but resize to a max width and height.

    Keyword arguments:
    img -- thei mage
    max_width -- maximum width
    max_height -- maximum height
    """    
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
    """Helper function to display. It will display an image and resize it
    to keep it below a certain width and height.

    Keyword arguments:
    img -- the image to display
    title -- the title for the window
    width -- the max width
    height -- the max height
    delay -- how long to wait for a key
    """    
    img = resize(img, width, height)            
    cv2.imshow(title, img)
    cv2.waitKey(delay)

def display_channels(img):
    """Helper function to display the separate color channels

    Keyword arguments:
    img -- the image to display
    """
    cv2.imshow("Red", img[:,:,2])
    cv2.imshow("Green", img[:,:,1])
    cv2.imshow("Blue", img[:,:,0])
    cv2.waitKey(1)

def generate_grid(imgs, grid_width, grid_height, labels = None):
    """Helper function to generate a grid of images. It is currently
    hard coded to display a grid of 2 images wide by three images high (6 images)
    
    It uses the grid_width and grid_height as a maximum for ALL images

    Keyword arguments:
    img -- the image to display
    grid_width -- the max width of the entire grid
    grid_height -- the max height of the entire grid
    """
    width = 1800
    height = 800
    
    new_img = np.zeros((height, width, 3), np.uint8)

    # always a grid of 2
    img_width = width // grid_width
    img_height = int(height // grid_height)
    
    x = 0
    y = 0
    for i, img in enumerate(imgs):
        ri = resize(img, img_width, img_height)
        if len(ri.shape) == 2:
            # Convert to color
            ri = cv2.cvtColor(ri, cv2.COLOR_GRAY2BGR)
        xstart = x * img_width
        ystart = y * img_height
        new_img[ystart:ystart + ri.shape[0], xstart:xstart + ri.shape[1], :] = ri
        if labels is not None:
            print("i: {:} -- x/y: {:}/{:} -- label: {:}".format(i, xstart, ystart, labels[i]))
            cv2.putText(new_img, labels[i], (xstart + 10, ystart + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

        x += 1
        if x >= grid_width:
            x = 0
            y += 1
            
    return new_img
   
def crop_blue(img, debug=False):
    """This function takes an image, tries to the find the bright blue
    area, crops around that area and rotates the image
    
    Keyword arguments:
    img -- the image to display
    debug -- True to display debug info
    """

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

def plot_contours(img, area_limit, arc_limit, aspect_limit, delay = 1):
    """This function takes plots the detected contours one at a time
    
    Keyword arguments:
    img -- the image to display
    area_limit - the min and max area limit
    arc_limit -- the min and max arc
    aspect_limit -- the min and max aspect ratio
    delay -- how long to delay between each contour plot
    """

    image, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        
    for i, c in enumerate(contours):
        cnt = cv2.approxPolyDP(c, 1, True)
        area = cv2.contourArea(cnt)
        arc = cv2.arcLength(cnt, True)
        
        if (area > area_limit[0]) and (area < area_limit[1]) and (arc > arc_limit[0]) and (arc < arc_limit[1]):
            mom = cv2.moments(cnt, True)
            cx = mom['m10'] / mom['m00']
            cy = mom['m01'] / mom['m00']
            r = cv2.boundingRect(cnt)
            aspect = r[2] / r[3]
            if aspect > 1:
                aspect = 1 / aspect
            if (aspect > aspect_limit[0]) and (aspect < aspect_limit[1]):
                print('Contour: {:} -- area: {:0.1f} -- arc: {:0.1f} -- cx: {:0.1f}, cy: {:0.1f}, aspect: {:}'.format(i, area, arc, cx, cy, aspect))
    
                temp = np.random.rand(3) * 255            
                color = (int(temp[0]), int(temp[1]), int(temp[2]))
                draw_img = cv2.drawContours(draw_img, [cnt], 0, color, -1)
                txt = str(i)            
                cv2.putText(draw_img, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    display(draw_img, 'Contours', delay = delay)


def process_contours(img, area_limit, arc_limit, aspect_limit, debug = False):
    """This function finds the contours and returns what it things are full digits
    
    Keyword arguments:
    img -- the image to display
    area_limit - the min and max area limit
    arc_limit -- the min and max arc
    aspect_limit -- the min and max aspect ratio
    debug -- display debug data
    """

    digit_cntrs = []
    image, contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)
    draw_img = np.zeros(img.shape, np.uint8)
    for i in range(len(contours)):
        cnt = cv2.approxPolyDP(top_cntrs[i], 1, True)
        area = cv2.contourArea(cnt)
        arc = cv2.arcLength(cnt, True)
        
        if (area > area_limit[0]) and (area < area_limit[1]) and (arc > arc_limit[0]) and (arc < arc_limit[1]):
            mom = cv2.moments(cnt, True)
            cx = mom['m10'] / mom['m00']
            cy = mom['m01'] / mom['m00']
            r = cv2.boundingRect(cnt)
            aspect = r[2] / r[3]
            if aspect > 1:
                aspect = 1 / aspect
            if (aspect > aspect_limit[0]) and (aspect < aspect_limit[1]):
                digit_cntrs.append(cnt)
                draw_img = cv2.drawContours(draw_img, [cnt], 0, 255, -1)


    morph_kernel = 25
    morph = cv2.morphologyEx(draw_img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel , morph_kernel)))
    image, contours, heirarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for c in contours:
        r = cv2.boundingRect(c)
        d = draw_img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        digits.append(d)
   
    if debug:
        plot_contours(img, area_limit, arc_limit, aspect_limit)
            
    return draw_img, digits

def find_connected_lines(img, threshold = 80, min_length = 30, gap = 10, debug = False):
    """This function is designed to connect lines that may be connected
    
    Keyword arguments:
    img -- the image to display
    threshold -- The threshold to use finding hough lines
    min_length -- the minimum length of each line
    gap -- the minimum length between lines
    delay -- how long to delay between each contour plot
    """

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
    """The backlight of the scale isn't uniform. This function applies
    a correction factor to normalize the brightness across the image
    
    Keyword arguments:
    img -- the image to display
    """
    
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

def test_thres():
    blur_amt = 7
    sigmaX = 0.9
    sigmaY = 0.9
    sigmaColor = 0.5
    sigmaSpace = 0.5
    block_size = 5
    C = -2
    morph_kernel = 9
    thres1 = 25
    thres2 = 50

    img = cv2.imread('../data/scale_13.9.png')
    
    cropped = crop_blue(img)
    gray = cropped[:, :, 1]
    correction = apply_correction(gray)
    correction = gray
    blur1 = cv2.blur(correction, (blur_amt, blur_amt))
    blur2 = cv2.medianBlur(correction, blur_amt)
    blur3 = cv2.GaussianBlur(correction, (blur_amt, blur_amt), sigmaX, sigmaY)
    blur4 = cv2.bilateralFilter(correction, blur_amt, sigmaColor, sigmaSpace)

    t1 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    ret, t2 = cv2.threshold(blur2, 50, 255, cv2.THRESH_BINARY_INV)
    #t2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    #t3 = cv2.Canny(blur, thres1, thres2)

    morph = cv2.morphologyEx(t2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel , morph_kernel)))


    imgs = [cropped, blur2, t1, t2, morph]
    debug_img = generate_grid(imgs, 1600, 900)
    display(debug_img, "debug", width = 1600, height=900)



def preprocess_image(img, debug = False):
    """This function takes an RGB image and process it into individual digits
    
    Keyword arguments:
    img -- the image to display
    delay -- how long to delay between each contour plot
    """

    blur_amt = 7
    bw_threshold = 50
    
    cropped = crop_blue(img)
    gray = cropped[:, :, 1]
    blur = cv2.medianBlur(gray, blur_amt)
    ret, thres = cv2.threshold(blur, bw_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # This produced nice contour images
    #plot_contours(morph_image, 4000, 100, 1)
    filled_image, digits = process_contours(thres, (900, 10000), (100, 800), (0, 1), False)
    
    rect_digits = []
    for d in digits:
        r = cv2.boundingRect(d)
        d = d[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        rect_digits.append(d)

    if debug:
        imgs = [cropped, gray, blur, thres, filled_image]
        debug_img = generate_grid(imgs, 2, 3)
        display(debug_img, "debug", width = 1600, height=900)
    return rect_digits
       
def create_training_data(cap, take_every_n_frame = 30, train_width = 10, train_height = 40):
    """This function creates a data set for the machine learning system
    
    Keyword arguments:
    cap -- a capture object with a loaded and valid movie
    take_every_n_frame -- Only take an image every N images
    train_width -- The max width of the training image
    train_height -- The max height of the training image
    """

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
                response_data.append(key)
                small = resize(d, train_width, train_height)
                #small = small.reshape((small.shape[0] * small.shape[1]))
                #small = np.concatenate((small, np.zeros(train_width * train_height - small.shape[0])))
                train_data.append(small)
                print('Inserted training data')
    
   
    np.save('train_data', train_data)
    np.save('response_data', response_data)
    
def train():
    """This function takes the training data and creates a working ML model
    """

    train_data = np.load('train_data.npy')
    response_data = np.load('response_data.npy')
    train_width = 10
    train_height = 40
    
    # Convert the response data into one hot encoding
    y_train = np.zeros((len(response_data), 10), np.float32)
    response_data -= ord('0')
    for i, v in enumerate(response_data):
        y_train[i, v] = 1        

    x_train = []
    for t in train_data:
        t = t.reshape((t.shape[0] * t.shape[1]))
        t = np.concatenate((t, np.zeros(train_width * train_height - t.shape[0])))
        x_train.append(t)

    x_train = np.array(x_train)
    x_train = x_train.astype('float32')    
    response_data = response_data.astype('float32')

    x_train /= 255

    model = cv2.ml.KNearest_create()
    #model = cv2.ml.SVMSGD_create()
    #model = cv2.ml.ANN_MLP_create()
    model.train(x_train, cv2.ml.ROW_SAMPLE, response_data)

    model.save('model')

    return model
    
def ml_test(model, digits, train_width = 10, train_height = 40, debug = False):
    """This function takes the model and the digits and returns the 
    detected numbers
    
    Keyword arguments:
    model -- The ML model
    digits -- an array of detected digits
    train_width -- The max width of the training image
    train_height -- The max height of the training image
    """

    p = []

    for d in digits:
        small = resize(d, train_width, train_height)
        small = small.reshape((small.shape[0] * small.shape[1]))
        small = np.concatenate((small, np.zeros(train_width * train_height - small.shape[0])))
        small = small.astype('float32')
        p.append(small)

    p = np.array(p)
    ret, results = model.predict(p)
    
    if debug:
        print('Predict ret: {:} -- results: {:}'.format(ret, results))
    
    num = 0
    for r in results:
        num *= 10
        num += r[0]
        
    return num

def hu_numbers():
    numbers = np.load('numbers.npy')
    for n in numbers:
        mom = cv2.moments(n)
        hu = cv2.HuMoments(mom)
        print(hu)        

def run_training():
    cap = cv2.VideoCapture('../data/test.mp4')        
    if cap.isOpened():
        create_training_data(cap)
        model = train()

def run():
    cap = cv2.VideoCapture('../data/test.mp4')        
    model = train()
    while cap.isOpened():
        ret, img = cap.read()
        while ret:
            ret, img = cap.read()
            digits = preprocess_image(img, True)
            num = ml_test(model, digits, debug = True)
            print('Detected number: {:}'.format(num))
            time.sleep(0.1)


#run_training()
run()    
#test_thres()
print('Hello')
