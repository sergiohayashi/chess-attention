import cv2


def write_image(path, img):
    cv2.imwrite(path, img)


def load_image_(path):
    # return cv2.imread( path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)


def write_label(path, text):
    f = open(path, "w")
    f.write(text)
    f.close()

def read_label( path):
    f= open( path)
    d= f.read()
    f.close()
    return d

def read_label_for( img_path):
    return read_label( img_path.replace( '.jpg', '.pgn').replace( 'images', 'labels'))
