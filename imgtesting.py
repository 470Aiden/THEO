import cv2




while True:
    cam = cv2.VideoCapture(0)
    height, width = cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH
    window = cv2.namedWindow("Camera",0)
    cc = cv2.CAP_FFMPEG
    cv2.VideoWriter("output.mp4",cc,20.0,(height,width),True)


# target = datetime.datetime.now() + datetime.timedelta(seconds=10)
# while True:
#     if datetime.datetime.now() == target:
#         print("Remember to take the chicken out the freezer")
#         target = datetime.datetime.now() + datetime.timedelta(seconds=10)

