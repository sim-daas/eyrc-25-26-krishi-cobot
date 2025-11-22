
import cv2
import cv2.aruco as aruco
import sys
import os

def detect_marker_in_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Common dictionaries to try
    dicts_to_try = [
        aruco.DICT_4X4_50,
        aruco.DICT_5X5_50,
        aruco.DICT_6X6_50,
        aruco.DICT_ARUCO_ORIGINAL
    ]
    
    dict_names = [
        "DICT_4X4_50",
        "DICT_5X5_50",
        "DICT_6X6_50",
        "DICT_ARUCO_ORIGINAL"
    ]

    print(f"--- Analyzing {os.path.basename(image_path)} ---")
    
    found = False
    for i, d_id in enumerate(dicts_to_try):
        try:
            aruco_dict = aruco.getPredefinedDictionary(d_id)
            parameters = aruco.DetectorParameters()
            
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None:
                print(f"  Detected with {dict_names[i]}: IDs {ids.flatten()}")
                found = True
        except Exception as e:
            print(f"  Error with {dict_names[i]}: {e}")

    if not found:
        print("  No markers detected with common dictionaries.")
    print("")

if __name__ == "__main__":
    # Paths provided by the user
    paths = [
        '/home/ubuntu/githubrepos/eyrc-25-26-krishi-cobot/ebot_description/meshes/ebot_aruco_tag.jpg',
        '/home/ubuntu/githubrepos/eyrc-25-26-krishi-cobot/eyantra_warehouse/models/fertiliser_can/meshes/aruco_tag_3.jpg'
    ]
    
    for p in paths:
        detect_marker_in_image(p)
