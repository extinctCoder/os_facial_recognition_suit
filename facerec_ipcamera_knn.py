
import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import numpy as np


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = []
    y = []
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in face_recognition.image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:

                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_face_locations = face_recognition.face_locations(X_frame)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    del draw
    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("train_data", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    process_this_frame = 29
    print('Setting cameras up...')
    cap = cv2.VideoCapture(0)
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
