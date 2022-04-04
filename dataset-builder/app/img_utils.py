import cv2


def resize( img, width, height):
    img= cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return img

