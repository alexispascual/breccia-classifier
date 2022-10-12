import cv2
import numpy as np

from sklearn.cluster import KMeans


def increase_constrast(image):

    # converting to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def main():

    image_file = './data/RI 05 026_b-trimmed.tif'
    n_clusters = 4

    image = cv2.imread(image_file)
    image = increase_constrast(image)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image_width = image.shape[0]
    image_height = image.shape[1]
    channels = image.shape[2]

    image = np.reshape(image, (image_width * image_height, channels))

    kmeans = KMeans(n_clusters, verbose=True).fit(image)

    background = [0 if c else 1 for c in (kmeans.labels_ == 0)]
    matrix = [0 if c else 1 for c in (kmeans.labels_ == 1)]
    clast_1 = [0 if c else 1 for c in (kmeans.labels_ == 2)]
    clast_2 = [0 if c else 1 for c in (kmeans.labels_ == 3)]

    # background = np.where(kmeans.labels_ == 0, blank_image, 0)
    # matrix = np.where(kmeans.labels_ == 1, blank_image, 0)
    # clast_1 = np.where(kmeans.labels_ == 2, blank_image, 0)
    # clast_2 = np.where(kmeans.labels_ == 3, blank_image, 0)

    background = np.reshape(background, (image_width, image_height))
    matrix = np.reshape(matrix, (image_width, image_height))
    clast_1 = np.reshape(clast_1, (image_width, image_height))
    clast_2 = np.reshape(clast_2, (image_width, image_height))

    print("Saving images...")
    cv2.imwrite("./data/background.png", background * 255)
    cv2.imwrite("./data/matrix.png", matrix * 255)
    cv2.imwrite("./data/clast_1.png", clast_1 * 255)
    cv2.imwrite("./data/clast_2.png", clast_2 * 255)


if __name__ == '__main__':
    main()
    print("====================== Done! ======================")
