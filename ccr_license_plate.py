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
        # store the minimum and maximum rectangular aspect ratio
        # values along with whether or not we are in debug mode
        self.minAR = min_aspect_ratio
        self.maxAR = max_aspect_ratio
        self.debug = debug

    def debug_show(self, title, image, wait_key=True):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        # at first we are different making morphological operations on the image

        # looking for dark elements on light backgrounds with size of license plate
        # to make them more distinct (MORPH_BLACKHAT option)
        rectKern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_show("After morph blackhat", blackhat)

        # Adding additional filter (X Sobel) to make background darker
        blackhat = blackhat / 255
        sobel = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
        sobel = sobel / 4
        gradX = abs(convolve(blackhat, sobel))
        gradX *= 255
        gradX = gradX.astype("uint8")
        self.debug_show("After sobel x", gradX)

        # applying blur and again doing morphological operation and creating another threshold
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        self.debug_show("After sobel x", gradX)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

        thresh = basic_threshold(gradX)
        # thresh = cv2.threshold(gradX, 0, 255,
        #                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_show("After basic threshold", thresh)

        # doing erosions and dilations to get rid of small white areas
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_show("After erode and dilate", thresh)

        # finding biggest white areas in the image (by their contours)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        # return the list of contours
        return cnts

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
