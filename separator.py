import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from PIL import Image
movement_threshold = 25  # Number of movements required to trigger an email
time_threshold = 10  # Time threshold in seconds for continuous motion
movement_counter = 0
last_notification_time = time.time()
donel = False
doner = False


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
    smtp_server.login('theftdetector84@gmail.com', 'yhfepxqoyswoiatn')  # Replace 'your_password' with the actual password
    smtp_server.sendmail('theftdetector84@gmail.com', 'harikiran707@gmail.com', msg.as_string())
    smtp_server.quit()

def detect_objects(frame):
    model = YOLO('models/best.pt')  # load an official model
    results = model.predict(frame)

    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
          
    img = annotator.result()  
    return img

log_file = open("motion_detection_log.txt", "a")

def select(event, x, y, flags, param):
    global x1, y1, x2, y2, donel, doner
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        donel = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        x2, y2 = x, y
        doner = True
        print(doner, donel)

def getFaceBox(frame):
    # Load pre-trained models
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    ageProto = "models/age_deploy.prototxt"
    ageModel = "models/age_net.caffemodel"
    genderProto = "models/gender_deploy.prototxt"
    genderModel = "models/gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Add age and gender labels
            face = frame[y1:y2, x1:x2]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]
            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]
            if age in ['(0-2)', '(4-6)', '(8-12)']:
                age = 'Child (0-18)'
            elif age in ['(15-20)', '(25-32)', '(38-43)']:
                age = 'Adult (20-40)'
            else:
                age = 'Elderly (50+)'
            labelGender = "Gender: " + gender
            labelAge = "Age: " + age
            cv2.putText(frame, labelGender, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, labelAge, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def check_motion(use_time_threshold=True):
    global x1, y1, x2, y2, donel, doner, movement_counter, last_notification_time
    cap = cv2.VideoCapture(0)

    # Set a lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("select_region")
    cv2.setMouseCallback("select_region", select)

    # Create a background subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        _, frame = cap.read()
        cv2.imshow("select_region", frame)

        if cv2.waitKey(1) == 27 or doner:
            cv2.destroyAllWindows()
            break

    while True:
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        frame1only = frame1[y1:y2, x1:x2]
        frame2only = frame2[y1:y2, x1:x2]

        diff = cv2.absdiff(frame2only, frame1only)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                cv2.putText(frame1, "Motion Detected", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                motion_detected = True
                break

        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 1)
         
        cv2.imshow("esc. to exit", frame1)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

        if motion_detected:
            movement_counter += 1
            print(movement_counter)
            current_time = time.time()

            if use_time_threshold:
                if current_time - last_notification_time >= time_threshold:
                    
                    frame1 = detect_objects(frame1)
                    frame1 = getFaceBox(frame1)
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    
                    send_email(frame_bytes)
                    last_notification_time = current_time  # Update the notification time

                    log_file.write(f"Motion detected at {time.ctime()} - Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")
                    

            else:
                if movement_counter >= movement_threshold:
                    frame1 = detect_objects(frame1)
                    frame1 = getFaceBox(frame1)
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    
                    send_email(frame_bytes)
                    movement_counter = 0  # Reset the counter after sending the email

                    log_file.write(f"Motion detected at {time.ctime()} - Movement Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")
                    

    log_file.close()  # Close the log file when finished
check_motion(use_time_threshold=True)
