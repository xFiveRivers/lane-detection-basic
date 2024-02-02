import cv2
import numpy as np


def canny(img, kernal=(5, 5), low_thresh=25, upper_thresh=100):
    """Applies Canny edge detection to an image

    Performs three image transformation operations. Frist,
    convert the image to grayscale, next apply Gaussian blue,
    and finally apply Canny edge detection

    Parameters
    ----------
    img : array_like
        The input image as loaded by OpenCV
    kernal : tuple, optional
        The kernal size for Gaussian blur, by default (5, 5)
    low_thresh : int, optional
        Lower theshold for Canny edge detection, by default 25
    upper_thresh : int, optional
        Upper threshold for Canny edge detection, by default 100

    Returns
    -------
    canny : array_like
        The resulting image after apply Canny edge detection
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, low_thresh, upper_thresh)

    return canny


def roi(img, roi_points):
    """Applies a mask to the image to only display a specified region
    
    Specifies a region of interest of the Canny image in which to display
    the lane lines by applying a black mask to all points outside the
    supplied polygon points.

    Parameters
    ----------
    img : array_like
        The ouput after Canny edge detection
    roi_points : array_like
        An array of the x and y vertices for the mask polygram  
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])

    Returns
    -------
    masked_img : array_like
        The image with a mask applied to the region of interest
    """

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi_points, 255)
    masked_img = cv2.bitwise_and(img, mask)

    return masked_img


def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
    """Draw Hough lines on supplied image

    Parameters
    ----------
    img : array_like
        The output after masking for the ROI
    lines : array_like
        An array containing the the starting and ending coordinates
        of the lines to draw.
        Example: np.array([x1, y1, x2, y2])
    color : list, optional
        RGB values for line color, by default [255, 0, 0]
    thickness : int, optional
        Thickness of drawn lines, by default 5
    """
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def get_coords(img, line, roi_points):
    """_summary_

    Parameters
    ----------
    img : array_like
        The output after masking for the ROI 
    line : array_like
        An array containing the slope and intercept of a line
        Example: np.array([slope, intercept])
    roi_points : array_like
        An array of the x and y vertices for the mask polygram  
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])

    Returns
    -------
    array_like
        The starting and ending coordinates of the line to be drawn
        in the region of interest
    """

    avg_slope, avg_int = line[0], line[1]
    y1 = img.shape[0]
    y2 = int(roi_points[0][1][1])
    x1 = int((y1 - avg_int) / avg_slope)
    x2 = int((y2 - avg_int) / avg_slope)

    return np.array([x1, y1, x2, y2])


def averaged_lines(img, lines, roi_points):
    """Finds the average of the left and right lines

    Finds the average line of the Hough lines detected for the left
    and right lane boundries

    Parameters
    ----------
    img : array_like
        The output after masking for the ROI 
    lines : array_like
        The output of the the Hough line transform
    roi_points : array_like
        An array of the x and y vertices for the mask polygram  
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])

    Returns
    -------
    array_like
        An array containing the starting and ending coordinates of the
        averaged left and right lines respectively
        Example: np.array([[x1L, y1L, x2L, y2L], [[x1R, y1R, x2R, y2R]]])
    """

    left_params = []
    right_params = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0], parameters[1]
        if slope < 0:
            left_params.append((slope, intercept))
        else:
            right_params.append((slope, intercept))

    avg_left = np.average(left_params, axis=0)
    avg_right = np.average(right_params, axis=0)

    left_coords = get_coords(img, avg_left, roi_points)
    right_coords = get_coords(img, avg_right, roi_points)

    return np.array([left_coords, right_coords])


def hough_lines(img, rho, theta, thresh, min_len, max_gap, roi_points):
    """Draws the averaged Hough lines for the left and right lane markers

    Parameters
    ----------
    img : array_like
        The output after masking for the ROI 
    rho : float
        Resolution of the rho parameter in pixels
    theta : float
        Resolution of the theta parameter in radians
    thresh : int
        Minimum number of intersections to detect lines
    min_len : _type_
        Minimum segment length to consider it as a line
    max_gap : _type_
        Maximum gap between segments to link them
    roi_points : array_like
        An array of the x and y vertices for the mask polygram  
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])

    Returns
    -------
    line_img : array_like
        The roi image with the averaged Hough lines drawn on them
    """
    lines = cv2.HoughLinesP(
        img, 
        rho, 
        theta, 
        thresh, 
        np.array([]), 
        minLineLength = min_len, 
        maxLineGap = max_gap
    )
    # print(f'Hough Lines \n {lines}')
    avg_lines = averaged_lines(img, lines, roi_points)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, avg_lines)

    return line_img


def weighted_image(orig_img, hough_img, alpha, beta, gamma=0):
    """Finds the weighted sum of two images

    Parameters
    ----------
    orig_img : array_like
        The original unmodified image
    hough_img : array_like
        The output image after applying Hough transforms
    alpha : float
        Weight of the original image
    beta : float
        Weights of the Hough image
    gamma : float, optional
        A scalar value added to each sum

    Returns
    -------
    array_like
        The averaged weighted image sum of the two input images
    """
    return cv2.addWeighted(orig_img, alpha, hough_img, beta, gamma)


def pipeline(img, roi_points):
    """_summary_

    Parameters
    ----------
    img : array_like
        Original image to be transformed
    roi_points : array_like
        An array of the x and y vertices for the mask polygram  
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])

    Returns
    -------
    array_like
        Resulting image with highlighted lane lines
    """

    canny_img = canny(np.copy(img))
    roi_canny_img = roi(canny_img, roi_points)
    hough_img = hough_lines(roi_canny_img, 2, np.pi/180, 50, 10, 5, roi_points)
    result = cv2.addWeighted(np.copy(img), 0.75, hough_img, 1.0, 0)

    return result


def draw_roi(img, points, color):
    """Draws the boundry of a region of interest on the image

    Parameters
    ----------
    img : array_like
        The image with the boundried to be drawn on
    points : array_like
        An array containing the the x and y coordinates of the
        vertices for the boundry
        Example: np.array([[(x1, y1), (x2, y2), (x3, y3)]])
    color : tuple
        A tuple containing the RGB values (respectively) for the boundry
        color

    Returns
    -------
    result : array_like
        The original image with the drawn ROI boundries
    """
    for point in points:
        for x, y in point:
            cv2.circle(img, (x, y), radius = 10, color = color, thickness = -1)

    result = cv2.polylines(img, points, True, color, thickness=5)

    return result