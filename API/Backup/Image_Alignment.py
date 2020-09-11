# =============================================================================
# This script was written for the alignament of the image with the reference image
# =============================================================================

#Import libraries

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import cv2
import os


def Image_Alignment(reference, skew_image,method = "AKAZE" ,save_match = False, save= False):
    if method == "AKAZE":
        print("Alignement is in process using '{}' method".format(method))        
        img1 = cv.imread(reference, cv.IMREAD_GRAYSCALE)  # referenceImage
        img2 = cv.imread(skew_image, cv.IMREAD_GRAYSCALE)  # sensedImage
    
        # Initiate AKAZE detector
        akaze = cv.AKAZE_create()
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = akaze.detectAndCompute(img1, None)
        kp2, des2 = akaze.detectAndCompute(img2, None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append([m])   
        if save_match:
            # Draw matches
            img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite('matches_Akaze.jpg', img3)
            # =============================================================================
        # Image Warping
        # =============================================================================
        # Select good matched keypoints
        ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
        sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
        
        # Compute homography
        H, status = cv.findHomography(sensed_matched_kpts, ref_matched_kpts, cv.RANSAC,5.0)
        # Warp image
        warped_image = cv.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        if save:
#            cv.imwrite("Aligned_Images"+skew_image, warped_image)
            cv.imwrite(os.path.join("Aligned_Images",skew_image.split("\\")[-1]), warped_image)            
        return warped_image

    if method == "SURF":
#        reference =  "0_Sample_Form.jpg"
#        #Skewed = "0_Guard_ CMS1500 5.jpg"
#        #Skewed = "E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\DATA\\Sample\\Guard_ CMS1500 12\\page_0.jpg"
#        skew_image = "E:\\Tushar\\Projects\\Image_Analytics\\OCR\\Template_Based_OCR\\DATA\\Sample\\Guard_ CMS1500 2\\page_0.jpg"
#   
        print("Alignement is in process using '{}' method".format(method))        
        img1 = cv.imread(reference, cv.IMREAD_GRAYSCALE)  # referenceImage
        img2 = cv.imread(skew_image, cv.IMREAD_GRAYSCALE)  # sensedImage
        
        surf = cv2.xfeatures2d.SURF_create(400)
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                                  ]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                                  ]).reshape(-1, 1, 2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)    
        #    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        #    ss = M[0, 1]
        #    sc = M[0, 0]
        #    scaleRecovered = math.sqrt(ss * ss + sc * sc)
        #    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        #    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))
            im_out = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1], img1.shape[0]))
    
        
        else:
            print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))
#            matchesMask = None
        if save:
#            cv.imwrite("Aligned_"+skew_image, im_out)
            cv.imwrite(os.path.join("Aligned_Images",skew_image.split("\\")[-1]), im_out)                        
        return im_out
#        cv2.imwrite("Aligned_SURF_CMS1500.jpg",im_out)