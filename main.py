import cv2


def main():

    image_file = './data/RI 05 026_b-trimmed.tif'

    image = cv2.imread(image_file)

    cv2.imshow("Image", image)


if __name__ == '__main__':
    main()
