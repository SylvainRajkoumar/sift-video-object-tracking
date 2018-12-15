import cv2
import numpy as np

FLANN_INDEX_KDTREE = 0
DISTANCE_THRESHOLD = 0.6
TREES = 5

# Query = image de référence (livre) qui va servir pour la détection de point d'intérêt
# Train = image provenant de la webcam
class SiftFeaturesMatcher(object):

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.query_image = cv2.imread("query_images/astrophysics_book.jpg", cv2.IMREAD_GRAYSCALE)
        self.query_keypoints, self.query_descriptors = self.sift.detectAndCompute(self.query_image, None)
        self.train_descriptors, self.train_keypoints = None, None

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES)
        search_params = dict()
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def get_good_matches(self, image):
        grayframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.train_keypoints, self.train_descriptors = self.sift.detectAndCompute(grayframe, None)
        matches = None

        if not (self.train_descriptors is None) and not (self.train_keypoints is None):
            if len(self.train_keypoints) >= 2:
                matches = self.flann_matcher.knnMatch(self.query_descriptors, self.train_descriptors, k=2)
        
        good_matches = []
        if matches != None:
            for m, n in matches:
                if m.distance < DISTANCE_THRESHOLD*n.distance:
                    good_matches.append(m)

        return good_matches

    def compute_homography(self, good_matches):
        # Récupération de la position de tous les points d'intérêt valide de référence
        query_points = np.float32([self.query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # Récupération de la position de tous les points d'intérêt valide de l'image courante
        train_points = np.float32([self.train_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography_matrix, mask = cv2.findHomography(query_points, train_points, cv2.RANSAC, 5.0)

        height, width = self.query_image.shape
        points = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
        perspective_transform = []
        if not (homography_matrix is None):
            perspective_transform = cv2.perspectiveTransform(points, homography_matrix)

        return perspective_transform

    def draw_homography(self, image, perspective_transform):
        homography_result_image = cv2.polylines(image, [np.int32(perspective_transform)], True, (0, 255, 0), 3)

        return homography_result_image

    def set_query_image(self, image_path):
        image = cv2.imread(image_path)
        grayframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.train_keypoints, self.train_descriptors = self.sift.detectAndCompute(grayframe, None)