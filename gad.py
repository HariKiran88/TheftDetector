import cv2


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



# Load input image
image_path = "kid1.jpg"
frame = cv2.imread(image_path)

# Call getFaceBox function
result_frame = getFaceBox(frame)

# Display result
cv2.imshow("Age-Gender Detection Result", result_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()