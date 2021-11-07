import pytesseract
import numpy as np
import imutils
import cv2.cv2 as cv2
from skimage.filters.edges import convolve

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def basic_threshold(img, tolerance=35):
    row = 0
    while row < len(img):
        element = 0
        while element < len(img[row]):
            if img[row][element] > tolerance:
                img[row][element] = 255
            else:
                img[row][element] = 0
            element += 1
        row += 1
    return img


def debug_show(name, image, debug=False):
    if debug:
        cv2.imshow(name, image)
        cv2.waitKey(0)


def ANPR(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debug_show("Gray", gray)

    # implementation of black hat morphology - closing (dilation after erosion) minus original image
    # kernel is the size of a typical license plate
    # it returns elements smaller than kernel and darker than their surroundings
    rectKern = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    closing = cv2.dilate(gray, rectKern)
    closing = cv2.erode(closing, rectKern)
    blackhat = closing - gray
    debug_show("Black hat", blackhat)

    # adding additional filter (X Sobel) to make background darker and vertical edges (which are more common in letters and numbers) more visible
    blackhat = blackhat / 255
    xSobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    xSobel = xSobel / 4
    xSobel = abs(convolve(blackhat, xSobel))
    xSobel *= 255
    xSobel = xSobel.astype("uint8")
    debug_show("Sobel X", xSobel)

    # applying basic blur to make contours of bright surfaces wider
    basicBlur = np.array(
        [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
    )
    blur = abs(convolve(xSobel, basicBlur))
    debug_show("Blur", blur)

    # applying basic threshold to make brighter object more distinct
    closing = cv2.dilate(blur, rectKern)
    closing = cv2.erode(closing, rectKern)
    thresh = basic_threshold(closing)
    debug_show("Threshold", thresh)

    # doing bunch of erosions and dilutions to get rid of small white areas
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    debug_show("After erode and dilate", thresh)

    # finding biggest white areas in the image (by their contours)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    licensePlateThreshold, licensePlateContour = None, None
    # we need to find a white spot that is rectangular enough to be license plate
    for contour in contours:
        xCord, yCord, width, height = cv2.boundingRect(contour)

        if 3 <= width / height <= 6:
            licensePlateContour = contour
            licensePlate = gray[yCord : yCord + height, xCord : xCord + width]
            licensePlateThreshold = basic_threshold(licensePlate, 150)

            debug_show("License Plate", licensePlate)
            debug_show("License Plate Threshold", licensePlateThreshold)
            break

    if licensePlateContour is not None:
        # reading license plate with tesseract
        # saying to tesseract that we are looking at one line of text consisting of only letters and numbers
        licensePlateText = pytesseract.image_to_string(
            licensePlateThreshold,
            config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7",
        )
    else:
        print("Error: Mo possible license plate candidates found in the image.")
        return None, None
    return licensePlateText, licensePlateContour


# import os
# path = '/Users/gkust/Desktop/ANPR/Photos/Back'
# files = os.listdir(path)
#
#
# for index, file in enumerate(files):
#     os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index + 1), '.jpg'])))

for fileNumber in range(20):
    imagePath = f"Photos/Back/{fileNumber + 1}.jpg"
    # loading and resizing image
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    # main function that returns license plate text and location
    licensePlateText, licensePlateContours = ANPR(image)
    if licensePlateText is not None and licensePlateContours is not None:
        # drawing contours on the original image
        box = cv2.boxPoints(cv2.minAreaRect(licensePlateContours))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 0, 255), 2)
        xCord, yCord, width, height = cv2.boundingRect(licensePlateContours)

        # deleting additional characters and white spaces
        licensePlateText = "".join(
            [c if ord(c) < 128 else "" for c in licensePlateText]
        ).strip()
        # displaying license text over it
        cv2.putText(
            image,
            licensePlateText,
            (xCord, yCord - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        print(imagePath, licensePlateText)
        cv2.imshow(f"Result {imagePath}", image)
        cv2.waitKey(0)
