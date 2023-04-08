import cv2

def resize_image(image, length):
    image = cv2.resize(image, (length, length), interpolation = cv2.INTER_AREA)
    return image
