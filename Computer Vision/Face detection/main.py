import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
 
    if cv2.waitKey(1) & 0xFF == ('q') or ret == False:
        break
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()