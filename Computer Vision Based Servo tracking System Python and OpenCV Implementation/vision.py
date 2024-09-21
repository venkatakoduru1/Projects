import cv2
import numpy as np
import serial
import time

ARDUINO_PORT = '/dev/ttyUSB0'
ARDUINO_BAUDRATE = 115200

MAX_SEND_RATE = 0.1  # seconds
CENTER_VAL = 1000  # value to send to indicate centering the servos

serial_port = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE)
print("Serial port opened")


def detect_ball(image):
    # convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of color in HSV
    # lower = np.array([10, 100, 100])
    # upper = np.array([20, 255, 255])
    # detect blue
    # lower = np.array([100, 100, 100])
    # upper = np.array([140, 255, 255])
    # detect green
    lower = np.array([40, 100, 100])
    upper = np.array([80, 255, 255])
    # threshold the HSV image to get only pixels within the range

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        return (int(x), int(y), int(radius))
    return None

def detect_face(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    if len(faces) > 0:
        face = faces[0]
        return (face[0] + face[2]//2, face[1] + face[3]//2, face[2]//2)


def draw_circle(image, ball):
    cv2.circle(image, (ball[0], ball[1]), ball[2], (0, 255, 0), 2)
    
def draw_center(image):
    height, width = image.shape[:2]
    cv2.line(image, (0, height//2), (width, height//2), (0, 0, 255), 2)
    cv2.line(image, (width//2, 0), (width//2, height), (0, 0, 255), 2)

def move_motors(ball, center, last_sent, last_detected):
    pan_difference = center[0] - ball[0]
    pan_val = pan_difference // (35 if abs(pan_difference) > 100 else 40)
    tilt_difference = ball[1] - center[1]
    tilt_val = tilt_difference // (35 if abs(tilt_difference) > 100 else 40)

    # if abs(pan_val) < 2:
    #     pan_val = 0
    # if abs(tilt_val) < 2:
    #     tilt_val = 0
    
    print(f"Pan: {pan_val}, Tilt: {tilt_val}")
    if time.time() - last_sent > MAX_SEND_RATE:
        last_sent = time.time()
        serial_port.write(f"{pan_val};{tilt_val}\n".encode())
    if time.time() - last_detected > 5:
        serial_port.write(f"{CENTER_VAL};{CENTER_VAL}\n".encode())


def main():
    cap = cv2.VideoCapture(0)
    last_sent = time.time()
    last_detected = time.time()
    while True:
        ret, frame = cap.read()
        # rotate frame 90 degrees clockwise
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # frame = cv2.flip(frame, 1)
        center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
        if not ret:
            break
        ball = detect_ball(frame)
        # ball = detect_face(frame)
        if ball is not None:
            draw_circle(frame, ball)
            # print(f"Ball position: {ball[0]}, {ball[1]}")
            last_detected = time.time()
            move_motors(ball, (center_x, center_y), last_sent, last_detected)
        else:
            move_motors((center_x, center_y), (center_x, center_y), last_sent, last_detected)
        draw_center(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            serial_port.write("f{CENTER_VAL};{CENTER_VAL}\n".encode())
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
