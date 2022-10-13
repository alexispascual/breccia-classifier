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


def erode_and_dilate(image):

    image = image.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel)

    return image


def create_overlay(image, overlay, name):

    _image = image.copy()
    _image = (_image - np.min(_image)) / (np.max(_image) - np.min(_image))

    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay = np.where(overlay == (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0))

    _image = cv2.addWeighted(_image, 0.8, overlay, 0.2, 0.0)
    cv2.imwrite(f"./data/results/overlayed_{name}.png", _image * 255)


def main():

    image_file = './data/images/RI 05 026_b-trimmed.tif'
    n_clusters = 4

    image = cv2.imread(image_file)
    image_copy = image.copy()
    image = np.where(image == (255, 255, 255), (0, 255, 0), image).astype(np.uint8)
    image = increase_constrast(image)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image_width = image.shape[0]
    image_height = image.shape[1]
    channels = image.shape[2]

    image = np.reshape(image, (image_width * image_height, channels))

    kmeans = KMeans(n_clusters, verbose=True).fit(image)

    class_0 = [0 if c else 1 for c in (kmeans.labels_ == 0)]
    class_1 = [0 if c else 1 for c in (kmeans.labels_ == 1)]
    class_2 = [0 if c else 1 for c in (kmeans.labels_ == 2)]
    class_3 = [0 if c else 1 for c in (kmeans.labels_ == 3)]

    class_0 = np.reshape(class_0, (image_width, image_height))
    class_1 = np.reshape(class_1, (image_width, image_height))
    class_2 = np.reshape(class_2, (image_width, image_height))
    class_3 = np.reshape(class_3, (image_width, image_height))

    class_0 = erode_and_dilate(class_0 * 255)
    class_1 = erode_and_dilate(class_1 * 255)
    class_2 = erode_and_dilate(class_2 * 255)
    class_3 = erode_and_dilate(class_3 * 255)

    print("Saving images...")
    cv2.imwrite("./data/results/class_0.png", class_0)
    cv2.imwrite("./data/results/class_1.png", class_1)
    cv2.imwrite("./data/results/class_2.png", class_2)
    cv2.imwrite("./data/results/class_3.png", class_3)

    print("Creating overlays...")
    create_overlay(image_copy, class_0, "class_0")
    create_overlay(image_copy, class_1, "class_1")
    create_overlay(image_copy, class_2, "class_2")
    create_overlay(image_copy, class_3, "class_3")

    print("Creating combined labels...")
    combined_labels = np.zeros((image_width * image_height, channels))
    for i, label in enumerate(kmeans.labels_):  # type: ignore
        match label:  # noqa: E999
            case 0:
                combined_labels[i] = (127, 135, 178)
            case 1:
                combined_labels[i] = (149, 218, 182)
            case 2:
                combined_labels[i] = (242, 230, 177)
            case 3:
                combined_labels[i] = (220, 133, 128)

    combined_labels = np.reshape(combined_labels, (image_width, image_height, channels))

    print("Saving combined labels...")
    cv2.imwrite("./data/results/combined_labels.png", combined_labels)

    print("Combining image and overlay...")
    image_copy = (image_copy - np.min(image_copy)) / (np.max(image_copy) - np.min(image_copy))
    combined_labels = (combined_labels - np.min(combined_labels)) / (np.max(combined_labels) - np.min(combined_labels))
    image_copy = cv2.addWeighted(image_copy, 0.8, combined_labels, 0.2, 0.0)    

    print("Saving combined overlay...")
    cv2.imwrite("./data/results/overlayed_all.png", image_copy * 255)


if __name__ == '__main__':
    main()
    print("====================== Done! ======================")
