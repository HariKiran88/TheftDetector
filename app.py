from flask import Flask, render_template, Response
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

donel = False
doner = False
x1, y1, x2, y2 = 0, 0, 0, 0
movement_counter = 0
last_notification_time = time.time()
log_file = open("motion_detection_log.txt", "a")

def send_email(image_data):
    global last_notification_time
    msg = MIMEMultipart()
    msg['Subject'] = 'Motion Detected'
    msg['From'] = 'theftdetector84@gmail.com'
    msg['To'] = 'theftmailcheck@gmail.com'

    image = MIMEImage(image_data, name='motion_detection.jpg')
    msg.attach(image)

    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login('theftdetector84@gmail.com', 'yhfepxqoyswoiatn')
    smtp_server.sendmail('theftdetector84@gmail.com', 'theftmailcheck@gmail.com', msg.as_string())
    smtp_server.quit()

def select(event, x, y, flags, param):
    global x1, y1, x2, y2, donel, doner
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        donel = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        x2, y2 = x, y
        doner = True

def rect_noise(use_motion_threshold, use_time_threshold, movement_threshold, time_threshold):
    global x1, y1, x2, y2, donel, doner, movement_counter, last_notification_time
    subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        _, frame1 = camera.read()
        _, frame2 = camera.read()

        frame1only = frame1[y1:y2, x1:x2]
        frame2only = frame2[y1:y2, x1:x2]

        diff = cv2.absdiff(frame2only, frame1only)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                cv2.putText(frame1, "Motion Detected", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                motion_detected = True
                break

        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imshow("esc. to exit", frame1)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

        if motion_detected:
            movement_counter += 1
            current_time = time.time()

            if use_time_threshold:
                if current_time - last_notification_time >= time_threshold:
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    send_email(frame_bytes)
                    last_notification_time = current_time

                    log_file.write(f"Motion detected at {time.ctime()} - Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")

            else:
                if movement_counter >= movement_threshold:
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    send_email(frame_bytes)
                    movement_counter = 0

                    log_file.write(f"Motion detected at {time.ctime()} - Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")

    log_file.close()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
