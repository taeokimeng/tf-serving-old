from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import json
import requests

model = VGG16(weights='imagenet')

img_path = 'lion.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

model.save('vgg16/2')

data = json.dumps({"instances": x.tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/vgg16/versions/2:predict',
                              data=data,
                              headers=headers)

predictions = json.loads(json_response.text)
print('Predicted:', decode_predictions(preds, top=3)[0])