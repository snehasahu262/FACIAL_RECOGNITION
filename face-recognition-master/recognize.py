import numpy as np
from sklearn.metrics import accuracy_score
from utils import  enhance_image

def recognize_face(embedding, embeddings, labels, threshold=0.5):
    distances = np.linalg.norm(embeddings - embedding, axis=1)
    
    argmin = np.argmin(distances)
    minDistance = distances[argmin]
    

    if minDistance>threshold:
        label = "Unknown"
    else:
        label = labels[argmin]

    return (label, minDistance)

if __name__ == "__main__":
    import cv2
    import argparse
    from detectors import detect_faces
    from extractors import extract_face_embeddings
    import cPickle
    import dlib

    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image", help="Path to image", required=True)
    ap.add_argument("-e","--embeddings", help="Path to saved embeddings",
                    default="/home/ubuntu/Downloads/snehafc/face-recognition-master/face_embeddings.npy")
    ap.add_argument("-l", "--labels", help="Path to saved labels",
                    default="/home/ubuntu/Downloads/snehafc/face-recognition-master/labels.cpickle")
    args = vars(ap.parse_args())

    embeddings = np.load(args["embeddings"])
    labels = cPickle.load(open(args["labels"]))
    shape_predictor = dlib.shape_predictor("/home/ubuntu/Downloads/snehafc/face-recognition-master/models/shape_predictor_5_face_landmarks.dat")
    face_recognizer = dlib.face_recognition_model_v1("/home/ubuntu/Downloads/snehafc/face-recognition-master/models/dlib_face_recognition_resnet_model_v1.dat")
    
    image = cv2.imread(args["image"])

    image_original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=enhance_image(image)

    faces = detect_faces(image)

    for face in faces:
        embedding = extract_face_embeddings(image, face, shape_predictor, face_recognizer)
        label = recognize_face(embedding, embeddings, labels)
        (x1, y1, x2, y2) = face.left(), face.top(), face.right(), face.bottom()
        #################################################################
        #facebolb=cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)
	#embedder.setInput(faceBlob)
	#vec = embedding.forward()
        #preds = recognizer.predict_proba(vec)[0]
	#j = np.rgmax(preds)
	#proba = preds[j]
	#name = le.classes_[j]
        #text = "{}: {:.2f}%".format(name, proba * 100)
        ##############################################################
        ####print("label[1]")
        #print(embedding)
        #print(embeddings)
        #print(labels)
        #####print(label[1])
        #print("accuracy score")
        #print(accuracy_score(label[1]))
        

        cv2.rectangle(image_original, (x1, y1), (x2, y2), (255, 120, 120), 2,cv2.LINE_AA)
        cv2.putText(image_original, label[0], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        print(label[0])

    #cv2.imshow("Image", image_original)
    cv2.waitKey(0)
