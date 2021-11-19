#Imports

import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


#Preprocessing the dataset images

from keras.preprocessing.image import ImageDataGenerator
Imagegen = ImageDataGenerator(rotation_range = 30,
                              width_shift_range = 0.2, 
                              height_shift_range = 0.2,
                              rescale = 1/255,
                              shear_range = 0.2,
                              zoom_range = 0.2,
                              horizontal_flip = True,
                              fill_mode = 'nearest')

train = Imagegen.flow_from_directory('face-mask-dataset/train', target_size = (150,150), batch_size = 20, class_mode = 'binary')
test = Imagegen.flow_from_directory('face-mask-dataset/test', target_size = (150,150), batch_size = 20, class_mode = 'binary')


#Building the model

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

model = Sequential([Conv2D(filters = 32, kernel_size = (3,3), input_shape = (150,150,3), activation = 'relu'),
                   MaxPooling2D((2,2)),
                   Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
                   MaxPooling2D((2,2)),
                   Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'),
                   MaxPooling2D((2,2)),
                   Flatten(),
                   Dense(128, activation = 'relu'),
                   Dense(1, activation = 'sigmoid')])
                   

#Compiling the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Training the model
model.fit_generator(train, epochs = 10, validation_data = test

#Real time face mask detection

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imwrite('test.jpg', frame)
    
    #face detection
    detector = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')
    face = detector.detectMultiScale(frame)
    
    from keras.preprocessing import image
    
    test_img = image.load_img('test.jpg', target_size = (150,150))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img = test_img/255
    
    if model.predict_classes(test_img)[0][0] == 1:
        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)
            cv2.putText(frame, 'Please put on a mask', (10,20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    else:
        for x, y, w, h in face:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0,), 3)
            cv2.putText(frame, 'You are safe', (10,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Face_mask_detector', frame)
    
    k = cv2.waitKey(10)
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows