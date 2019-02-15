import cv2
import sys

# Pass in the directory containing XML for cascades
cascPath = sys.argv[1]

# Createing a classifier
faceCascade = cv2.CascadeClassifier(cascPath)

video= cv2.VideoCapture(0)

while True:
    # Capturing frame-by-frame
    ret, frame = video.read()

    # Converting frame to grascale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # To drawing a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame 
    cv2.imshow('Video', frame)

    # To terminate the process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#releasing
video.release()
cv2.destroyAllWindows()
