import cv2
import numpy as np

def is_low_light(img_np, bbox, threshold=110):
    """
    Checks brightness only on face region
    """
    x1, y1, x2, y2 = bbox.astype(int)
    face = img_np[y1:y2, x1:x2]

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)

    return brightness < threshold, brightness


def is_blurry(img_np, bbox, threshold=100):
    """
    Variance of Laplacian for blur detection
    """
    x1, y1, x2, y2 = bbox.astype(int)
    face = img_np[y1:y2, x1:x2]

    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return blur_score < threshold, blur_score
