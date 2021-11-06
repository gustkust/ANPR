import pytesseract
import numpy as np
import imutils
import cv2.cv2 as cv2
from skimage.segmentation import clear_border
from skimage.filters.edges import convolve

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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


class PyImageSearchANPR:
    def __init__(self, min_aspect_ratio=3, max_aspect_ratio=6, debug=False):
        self.minAR = min_aspect_ratio
        self.maxAR = max_aspect_ratio
        self.debug = debug

    def debug_show(self, title, image, wait_key=True):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray):
        self.debug_show("Gray", gray)

        # implementation of black hat morphology - closing (dilation after erosion) minus original image
        # kernel is the size of a typical license plate
        # it returns elements smaller than kernel and darker than their surroundings
        rectKern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        closing = cv2.dilate(gray, rectKern)
        closing = cv2.erode(closing, rectKern)
        blackhat = closing - gray
        self.debug_show("Black hat", blackhat)

        # adding additional filter (X Sobel) to make background darker and vertical edges (which are more common in letters and numbers) more visible
        blackhat = blackhat / 255
        xSobel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
        xSobel = xSobel / 4
        xSobel = abs(convolve(blackhat, xSobel))
        xSobel *= 255
        xSobel = xSobel.astype("uint8")
        self.debug_show("Sobel X", xSobel)

        # applying basic blur to make contours of bright surfaces wider
        basicBlur = np.array([[1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9],
                             [1/9, 1/9, 1/9]])
        blur = abs(convolve(xSobel, basicBlur))
        self.debug_show("Blur", blur)

        # applying basic threshold to make brighter object more distinct
        closing = cv2.dilate(blur, rectKern)
        closing = cv2.erode(closing, rectKern)
        thresh = basic_threshold(closing)
        self.debug_show("Threshold", thresh)

        # doing bunch of erosions and dilutions to get rid of small white areas
        thresh = cv2.erode(thresh, None, iterations=5)
        thresh = cv2.dilate(thresh, None, iterations=5)
        self.debug_show("After erode and dilate", thresh)

        # finding biggest white areas in the image (by their contours)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours

    def locate_license_plate(self, gray, candidates,
                             clearBorder=False):
        lpCnt = None
        roi = None
        # we need to find first white spot that is rectangular enough
        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if self.minAR <= ar <= self.maxAR:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                if clearBorder:
                    roi = clear_border(roi)

                self.debug_show("License Plate", licensePlate)
                self.debug_show("ROI", roi, wait_key=True)
                break
        return (roi, lpCnt)

    def build_tesseract_options(self, psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        # initialize the license plate text
        lpText = None
        # convert the input image to grayscale, locate all candidate
        # license plate regions in the image, and then process the
        # candidates, leaving us with the *actual* license plate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates,
                                                clearBorder=clearBorder)
        # only OCR the license plate if the license plate ROI is not
        # empty
        if lp is not None:
            # OCR the license plate
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_show("License Plate", lp)
        # return a 2-tuple of the OCR'd license plate text along with
        # the contour associated with the license plate region
        return (lpText, lpCnt)


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# initialize our ANPR class
anpr = PyImageSearchANPR(debug=True)
# grab all image paths in the input directory
imagePath = "005.jpg"

# load the input image from disk and resize it
image = cv2.imread(imagePath)
image = imutils.resize(image, width=600)
# apply automatic license plate recognition
(lpText, lpCnt) = anpr.find_and_ocr(image)
# only continue if the license plate was successfully OCR'd
if lpText is not None and lpCnt is not None:
    # fit a rotated bounding box to the license plate contour and
    # draw the bounding box on the license plate
    box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
    box = box.astype("int")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    # compute a normal (unrotated) bounding box for the license
    # plate and then draw the OCR'd license plate text on the
    # image
    (x, y, w, h) = cv2.boundingRect(lpCnt)
    cv2.putText(image, cleanup_text(lpText), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # show the output ANPR image
    print("[INFO] {}".format(lpText))
    cv2.imshow("Output ANPR", image)
    cv2.waitKey(0)
