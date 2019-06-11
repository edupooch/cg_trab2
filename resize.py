from imutils import paths
import cv2

for imagePath in paths.list_images("data"):
    img = cv2.imread(imagePath, 0)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(imagePath, img)
