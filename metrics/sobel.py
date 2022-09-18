import cv2


def read_img_toYCRCB(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return img


def calculate_YCRCB_gradient(YCRCB_img):
    YCRCB_img_Y_channel = YCRCB_img[..., 0]
    sobelx = cv2.Sobel(YCRCB_img_Y_channel, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(YCRCB_img_Y_channel, -1, 0, 1, ksize=3)
    grad = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return grad


def calculate_grad_avg(grad):
    h, w = grad.shape
    return grad.sum() / h / w


def calculate_sobel_gradient_pipeline(img_path):
    img_ycrcb = read_img_toYCRCB(img_path)
    grad = calculate_YCRCB_gradient(img_ycrcb)
    grad_avg = calculate_grad_avg(grad)
    return grad, grad_avg
