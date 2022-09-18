import cv2


def read_img_toRGB(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calculate_histogram(
    image, channels=[0], hist_size=[10], hist_range=[0, 256]
):
    # convert to different color space if needed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image_hist = cv2.calcHist([image], channels, None, hist_size, hist_range)
    image_hist = cv2.normalize(image_hist, image_hist).flatten()
    return image_hist


def compare_images_histogram(img_base, img_compare, method="correlation"):
    hist_1 = calculate_histogram(img_base)
    hist_2 = calculate_histogram(img_compare)

    if method == "intersection":
        comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_INTERSECT)
    else:
        comparison = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_CORREL)
    return comparison


def compare_images_histogram_pipeline(img_base_path, img_compare_path):
    img_base = read_img_toRGB(img_base_path)
    img_compare = read_img_toRGB(img_compare_path)
    similarity = compare_images_histogram(img_base, img_compare)
    return similarity
