import cv2
import numpy as np

alphabet = list("abcdefghijklmnopqrstuvwxyz")
ordalphabet = [ord(i) for i in alphabet]  # buttons

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
lighting = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
check = 0

faceCascade = cv2.CascadeClassifier(r"C:\Users\zorin\PycharmProjects\facerecognition\venv\Lib\site-packages\cv2\data"
                                    r"\haarcascade_frontalface_default.xml")  # Location of frontalface_default
capture = cv2.VideoCapture(0)
capture.set(3, 800)  # set Width
capture.set(4, 600)  # set Height

while True:
    ret, img = capture.read()

    faces = faceCascade.detectMultiScale(  # find face
        img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # rectangle around faces
        roi_color = img[y:y + h, x:x + w]

    if check == 0:
        cv2.imshow("My picture", img)  # Default
    if check == 1:
        img = cv2.filter2D(img, -1, kernel)  # Sharp
        cv2.imshow("My picture", img)
    if check == 2:
        img = cv2.filter2D(img, -1, lighting)  # Light
        cv2.imshow("My picture", img)
    if check == 3:
        img = cv2.filter2D(img, -1, sobel)  # Dark
        cv2.imshow("My picture", img)

    key = cv2.waitKey(30) & 0xFF
    if key in ordalphabet:  # Any button to change mode
        check += 1
        if check == 4:
            check = 0
    if key == 27:   # Esc to exit
        break

capture.release()
cv2.destroyAllWindows()
