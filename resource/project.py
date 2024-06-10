import time
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

font = cv2.FONT_ITALIC
eyes_close = 2  # 눈이 감긴 상태 시간 설정 (2초)
before_eyes_close = None  # 눈을 감기전 시간을 기록하는 공간

def detect(gray, frame):
    global before_eyes_close
    #얼굴검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # 이미지 프레임에 (x,y)에서 시작, (x+넓이, y+높이)까지의 사각형을 그림
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 이미지를 얼굴 크기 만큼 잘라서 그레이스케일 이미지와 컬러이미지를 만듬
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 눈 검출
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        if len(eyes) > 0:
            # 눈을 뜬 상태
            before_eyes_close = time.time()
            # 눈 영역 표시
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(frame, "Open eyes", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)  # 눈을 찾았다는 메시지
        else:
            # 눈을 감은 상태
            if before_eyes_close is not None and (time.time() - before_eyes_close) >= eyes_close:
                cv2.putText(frame, "Wake Up! Closed Eyes Detected!", (5, 50), font, 1.0, (0, 0, 255), 2) # 눈을 감았다는 메시지
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = detect(gray, frame)

    cv2.imshow("eye detect", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

