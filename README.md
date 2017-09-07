# TensorFlow Estimator Examples

A simple digit captcha solver with TensorFlow Estimator DNNClassifier. 

## Requirements

* Python 2.7
* Linux
* tensorflow
* tensorflow_serving


Install modules with pip (and not pip3 as tensorflow_serving not available in python3 yet)

```bash
pip install tensorflow
pip install tensorflow_serving

echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server
```

## Running Examples

#### Setup

```bash
git clone https://github.com/panangad/tensorflow-estimator-examples
cd tensorflow-estimator-examples
```


#### Captcha solver Model Training and Export

```bash
python digit_captcha_solver_train.py
```


#### Starting Tensorflow Model Server
Open a new shell and start the server

```bash
tensorflow_model_server --port=9000 --model_name=classifier --model_base_path=/tmp/savedmodel
```

#### Running Captcha solver client
```bash
python digit_captcha_solver_client.py
```

