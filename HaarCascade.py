import urllib.request
import cv2
import numpy as np
import os


def store_raw_images():
    # http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152

    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 731

    if not os.path.exists('negative'):
        os.makedirs('negative')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "negative/" + str(pic_num) + ".mp4")
            img = cv2.imread("negative/" + str(pic_num) + ".mp4", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            # resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("negative/" + str(pic_num) + ".mp4", img)
            pic_num += 1

        except Exception as e:
            print(str(e))

def create_pos_n_neg():
    for file_type in ['pos']:

        for img in os.listdir(file_type):
            if file_type == 'negs':
                line = file_type+'/'+img+'\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)

            elif file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)

def findBoundingBoxes():
    face_cascade = cv2.CascadeClassifier('/Users/clarkpathakis/PycharmProjects/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    directory = "/Users/clarkpathakis/PycharmProjects/faceTracking/pos"
    for file in os.listdir(directory):
        gray = cv2.imread(directory+"/"+file, 0)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        is_null = True
        for (x, y, w, h) in faces:
            is_null = False
            line = directory+"/"+file+" 1 "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)+"\n"
            with open('info.dat', 'a') as f:
                f.write(line)
            with open('info.txt', 'a') as f:
                f.write(line)
        if is_null:
            with open('null_list.txt', 'a') as f:
                f.write(file+'\n')


def iphoneExample():

    pic_num = 0
    try:
        img = cv2.imread("/image" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # should be larger than samples / pos pic (so we can place our image on it)
        # resized_image = cv2.resize(img, (100, 100))
        cv2.imwrite("/" + str(pic_num) + ".jpg", img)
        pic_num += 1

    except Exception as e:
        print(str(e))

# iphoneExample()
# create_pos_n_neg()
findBoundingBoxes()
# store_raw_images()
