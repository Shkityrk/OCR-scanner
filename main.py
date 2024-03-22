import time

import cv2
import numpy as np

import os

from matplotlib import pyplot as plt

from constants import emnist_labels
from models import emnist_model

# Force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Debug messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

import idx2numpy


def cnn_print_digit(d):
    print(d.shape)
    for x in range(28):
        s = ""
        for y in range(28):
            s += "{0:.1f} ".format(d[28*y + x])
        print(s)


def cnn_print_digit_2d(d):
    print(d.shape)
    for y in range(d.shape[0]):
        s = ""
        for x in range(d.shape[1]):
            s += "{0:.1f} ".format(d[x][y])
        print(s)

def letters_extract(image_file: str):

    #предобработка изображений
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=2)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            print("letter_crop.shape", letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (28, 28), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    # cv2.imshow("Input", img)
    # cv2.imshow("Enlarged", img_erode)
    # cv2.imshow("Output", output)
    # for index in range(len(letters)):
    #     cv2.imshow(str(index), letters[index][2])
    #
    # cv2.waitKey(100)
    return letters



def training():
    # def emnist_train(model):
    model = emnist_model()
    emnist_path = 'lib/emnist_source_files'

    t_start = time.time()

    X_train = idx2numpy.convert_from_file(
        "lib/emnist_source_files/emnist-byclass-train-images-idx3-ubyte")
    y_train = idx2numpy.convert_from_file(
        "lib/emnist_source_files/emnist-byclass-train-labels-idx1-ubyte")

    X_test = idx2numpy.convert_from_file(
        "lib/emnist_source_files/emnist-bymerge-test-images-idx3-ubyte")
    y_test = idx2numpy.convert_from_file(
        "lib/emnist_source_files/emnist-bymerge-test-labels-idx1-ubyte")

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

    k = 10
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                                patience=3,
                                                                verbose=1,
                                                                factor=0.5,
                                                                min_lr=0.00001)

    model.fit(X_train,
              x_train_cat,
              validation_data=(X_test, y_test_cat),
              callbacks=[learning_rate_reduction],
              batch_size=64,
              epochs=30)

    print("Training done, dT:", time.time() - t_start)

    model.save('emnist_letters_old.h5')



def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])



    img_arr = img_arr.reshape((1, 28, 28, 1))



    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])

def img_to_str(model, image_file: str):
    letters = letters_extract(image_file)

    # for i in range(len(letters)):
    #     plt.subplot(1, len(letters), i + 1)
    #     plt.imshow(letters[i][2], cmap='gray')
    #     plt.axis('off')
    # plt.show()

    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out

def main():
    # letters_extract()

    # training()

    model = keras.models.load_model('emnist_letters_old.h5')
    s_out = img_to_str(model, "image.png")
    print(s_out)


if __name__ == "__main__":
    main()
