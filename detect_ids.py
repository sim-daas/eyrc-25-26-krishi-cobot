
import cv2
import cv2.aruco as aruco
import sys

def detect_marker(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try 4x4_50 first as hinted in other files
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        print(f"File: {image_path} -> IDs found: {ids.flatten()}")
    else:
        # Try 5x5 just in case
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is not None:
             print(f"File: {image_path} -> IDs found (5x5): {ids.flatten()}")
        else:
             print(f"File: {image_path} -> No IDs found")

if __name__ == "__main__":
    detect_marker('/home/ubuntu/githubrepos/eyrc-25-26-krishi-cobot/ebot_description/meshes/ebot_aruco_tag.jpg')
    detect_marker('/home/ubuntu/githubrepos/eyrc-25-26-krishi-cobot/eyantra_warehouse/models/fertiliser_can/meshes/aruco_tag_3.jpg')
