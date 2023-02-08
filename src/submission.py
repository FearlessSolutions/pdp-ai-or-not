import os
from datetime import date
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Avoid OOM errors by limiting GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# Directory paths to use
TEST_IMG_PATH = '../data/test/'
TEST_PREDICTIONS_PATH = '../data/submission.csv'
MODEL_PATH = '../models/'
img_height = 180
img_width = 180

def create_submission(model):
    
    df = pd.DataFrame(columns= ['id', 'label'])
    images = os.listdir(TEST_IMG_PATH)

    for image in images:
      img_path = TEST_IMG_PATH + image
      img = tf.keras.utils.load_img(img_path,
                                    target_size=(img_height, 
                                                 img_width))
      img_array = tf.keras.utils.img_to_array(img)
      img_array = np.array([img_array])

      pred = model.predict(img_array)
      score = tf.nn.softmax(pred[0])
      df2 = pd.DataFrame({'id': image}, {'label': score})
      df = df.append(df2, ignore_index=True)
    return df

model = load_model(MODEL_PATH + 'tf_model_20230203.h5')
df = create_submission(model)
df.to_csv('../data/'+str(date.today())+'_submission',index=False)