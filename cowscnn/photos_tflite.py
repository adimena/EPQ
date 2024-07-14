import os
from datetime import datetime
import time

from colorama import Fore
import glob
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

from keras.preprocessing import image


interpreter = tf.lite.Interpreter('./model.tflite')
interpreter.allocate_tensors()


resultsdict = {'timestamp': [], 'cows': []}

for i in range(100):
    timenow = datetime.now().isoformat()
    resultsdict['timestamp'].append(timenow)
    CMD = "fswebcam --device /dev/video0 -r 640x480 --jpeg 85 -D 1 photo.jpg"
    os.system(CMD)
    FNAME = "photo.jpg"
    
    img = image.load_img(FNAME, target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_index = np.argmax(output_data[0])
    result = output_data < 0.5
    
    images = np.vstack([x])
    #classes = model.predict(images)
    #result = classes[0][0] < 0.5
    resultsdict['cows'].append(result)
    
    if result:
        print(Fore.GREEN + 'COW DETECTED')
    else:
        print(Fore.RED + 'NO COW DETECTED')
    
    df = pd.DataFrame(resultsdict)
    df.to_csv('results3.csv', index=False)
    
    time.sleep(5)
