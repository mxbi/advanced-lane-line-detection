import numpy as np
import cv2
import matplotlib.image as mpimg

class CameraCalibration():
    """
    This class allows for the calibration of camera images. When intitialised, it takes in a list of images, as well as the size of the chessboard. 
    It then analyses each image in turn, and detects the chessboards on all these images. Once all the chessboard points have been found,
    it creates a camera and distortion matrix, allowing for this matrix to be applied to any image from the same camera by using CameraCalibration.undistort()
    """
    def __init__(self, imgs, hcount, vcount):
        self.mtx = None  # Camera Matrix
        self.dist = None  # Distortion Coefficients

        self.size = None  # Image size
        self.hcount = hcount  # Width of chessboard
        self.vcount = vcount  # Height of chessboard

        self.calibrated = False  # Whether the camera has been calibrated

        objpoints, imgpoints = self._get_all_points(imgs)
        self.mtx, self.dist = self._calibrate(objpoints, imgpoints)

    # Generate a matrix of x, y, z object points
    def _gridspace(self, ):
        # Initialise 2D array of zeros, needs to be float to stop OpenCV from complaining
        obj = np.zeros((self.hcount * self.vcount, 3)).astype(np.float32)
        # For x and y, fill with a grid of ascending values
        obj[:, :2] = np.mgrid[0:self.hcount, 0:self.vcount].T.reshape(-1, 2)
        return obj

    # Get the image points and the object points for a single image
    def _get_points(self, img):
        objp = self._gridspace() # Generate 3D object points for chessboard
        
        # Convert to grayscale, using RGB instead of BGR as mpimg uses a RGB format
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Save size of image for later use
        self.size = grey.shape[::-1]

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(grey, (self.hcount, self.vcount), None)
        
        # If OpenCV was able to find the chessboard, we return the corners along with 
        # the object points for calibration. Otherwise, we return None (when an error occurs)
        if ret:
            return corners, objp
        else:
            return None, None

    # Get the object and image points for all images.
    def _get_all_points(self, imgs):
        objpoints = []
        imgpoints = []
        for i, img in enumerate(imgs):
            imgp, objp = self._get_points(img)
            if imgp is None:
                # Warn if image is skipped
                print('[CameraCalibration] Image {} skipped during calibration as chessboard is not fully visible'.format(i))
            else:
                objpoints.append(objp)
                imgpoints.append(imgp)
            
        return objpoints, imgpoints

    # Get camera coefficients
    def _calibrate(self, objpoints, imgpoints):
        # Calculate the camera matrix and the distortion coefficients. This is what we need in order to undistort
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.size, None, None)
        return mtx, dist

    # Apply the distortion coefficients to an image
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)