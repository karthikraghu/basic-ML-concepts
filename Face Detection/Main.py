import cv2
import numpy as np
import matplotlib.pyplot as plt
from Eigenfaces import process_and_train, classify_image, create_database_from_folder, reconstruct_image
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from time import time
import glob
from sklearn.metrics import classification_report

# Set mode: True for webcam, False for dataset testing
onlineDetection = False

if onlineDetection:
    # webcam mode
    N = 64
    
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # load images from folder
    labels, train, num_images = create_database_from_folder(glob.glob('eigenfaces/*.png'))
    
    # train the system
    u, num_eigenfaces, avg = process_and_train(labels, train.T, num_images, N, N)
    
    # start camera
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Camera not found. Trying other indices...")
        for i in [1, 2, 3]:
            video_capture = cv2.VideoCapture(i)
            if video_capture.isOpened():
                print(f"Camera found on index {i}")
                break
        else:
            print("No camera available. Set onlineDetection = False to use dataset mode.")
            exit()

    print("Camera ready. Press 'q' to quit.")
    
    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to get frame")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(N - 10, N - 10))

        for (x, y, w, h) in faces:
            face = frame[y: y + h, x: x + w]
            image = gray[y: y + h, x: x + w]

            image = cv2.equalizeHist(image)
            image = cv2.resize(image, (N, N))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            start = time()
            prediction = classify_image(image, u, avg, num_eigenfaces, N, N)
            pred_value = prediction[0]
            
            processing_time = (time() - start) * 1000
            
            # get name from prediction
            try:
                if isinstance(pred_value, (int, float)):
                    pred_name = f"Person_{int(pred_value)}"
                else:
                    pred_str = str(pred_value).strip("'\"")
                    if "_" in pred_str:
                        pred_name = pred_str.split("_")[0]
                    else:
                        pred_name = pred_str
            except:
                pred_name = "Unknown"
            
            time_text = f"{int(processing_time)} ms"
            
            cv2.putText(frame, pred_name, (int(x), int(y + h + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, time_text, (int(x), int(y + h + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        try:
            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

    print("Closing camera...")
    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Done.")

else:
    # dataset mode
    lfw_dataset = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

    n_images, h, w = lfw_dataset.images.shape

    X = lfw_dataset.data
    y = lfw_dataset.target
    labels = lfw_dataset.target_names
    n_labels = labels.shape[0]

    print("Dataset info:")
    print("Images: ", format(n_images))
    print("People: ", format(n_labels))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    n_train_images = X_train.shape[0]
    n_test_images = X_test.shape[0]
    print("Training images: ", format(n_train_images))

    # train
    t0 = time()
    u, num_eigenfaces, avg = process_and_train(y_train, X_train, n_train_images, h, w)
    print("Training time: %0.3fs" % (time() - t0))

    # test reconstruction
    test_img = np.copy(X_test[0])
    reco_10 = reconstruct_image(np.copy(test_img), u, avg, 10, h, w)
    reco_100 = reconstruct_image(np.copy(test_img), u, avg, 100, h, w)
    reco_full = reconstruct_image(np.copy(test_img), u, avg, num_eigenfaces, h, w)

    plt.suptitle('Image Reconstruction')
    plt.subplot(1, 4, 1)
    plt.imshow(test_img.reshape(h, w), cmap='gray')
    plt.title('Original')

    plt.subplot(1, 4, 2)
    plt.imshow(reco_10, cmap='gray')
    plt.title('10 eigenfaces')

    plt.subplot(1, 4, 3)
    plt.imshow(reco_100, cmap='gray')
    plt.title('100 eigenfaces')

    plt.subplot(1, 4, 4)
    plt.imshow(reco_full, cmap='gray')
    plt.title('All eigenfaces')

    plt.axis('off')
    plt.show()

    # test classification
    t0 = time()

    predictions = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        pred = classify_image(X_test[i], u, avg, num_eigenfaces, h, w)
        predictions[i] = pred[0]

    test_time = (time() - t0)
    print("Testing time: %0.3fs for %d faces" % (test_time, n_test_images))
    print("Time per face: %0.3fs" % (test_time / n_test_images))

    print(classification_report(y_test, predictions, target_names=labels))
