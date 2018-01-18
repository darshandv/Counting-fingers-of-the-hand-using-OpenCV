#Counting fingers using OpenCV

OpenCV is an open-source library which helps in image processing and video analysis. This library is used to achieve my result. The entire code is written in python3.
Using the webcam the number of fingers is calculated for the captured frame with a hand.

The steps in the implementation :
1. The video is captured using the webcam.
1. The background is removed. This step basically eliminates all the things that are motionless and tries to capture the hand.
1. Then the whole frame is cropped (the top right part is the remain) for the easy analysis of the image.
1. It is then filtered and blurred , the largest contour is found , fingers are calculated for that contour.
1. Two video analysis - outputs are shown :
    1. The blurred image of the hand captured.
    1. The way the algorithm modifies it for calculation ( with contours drawn ).
1. The output is shown on the terminal.
