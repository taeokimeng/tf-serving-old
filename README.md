# TensorFlow Serving using Docker

Start TensorFlow serving server

~~~
docker run -t -p 8501:8501 --name tf_serving_vgg16 --mount type=bind,source=/home/tokim/code/tf-serving/vgg16/,target=/models/vgg16 tensorflow/serving --model_config_file=/models/vgg16/models.config --model_config_file_poll_wait_seconds=60
~~~
* REST API port 8501 to our host's port 8501
* Docker container name is tf_serving_vgg16
* Provide the model configuration file
* Check changes in the configuration file every 60 seconds