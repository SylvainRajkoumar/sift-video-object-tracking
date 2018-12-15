import cv2
import numpy as np
from utils.sift_features_matcher import SiftFeaturesMatcher

MIN_MATCHES_THRESHOLD = 20

class ObjectTracking(object):

    def __init__(self):
        self.sift_features_matcher = SiftFeaturesMatcher()

    def get_tracking_result_image(self, image):
        good_matches = self.sift_features_matcher.get_good_matches(image)
        # Calcul de l'homographie si il y a assez de descripteur/point d'intérêt similaire
        if len(good_matches) > MIN_MATCHES_THRESHOLD:
            perspective_transform = self.sift_features_matcher.compute_homography(good_matches)
            if len(perspective_transform) != 0:
                result_image = self.sift_features_matcher.draw_homography(image, perspective_transform)
                return result_image
        return image
