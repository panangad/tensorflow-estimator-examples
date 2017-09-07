from __future__ import print_function

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import glob2
import ntpath
from operator import attrgetter

from image_preprocess import get_arrays_from_img



numberlist = numpy.array(get_arrays_from_img('predict_sample.png'))
hostport = "localhost:9000"
host, port = hostport.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = classification_pb2.ClassificationRequest()
request.model_spec.name = 'classifier'
example = request.input.example_list.examples.add()
predicted_num = ''
for onenumber in numberlist:
  del(example.features.feature['x'].float_list.value[:])
  example.features.feature['x'].float_list.value.extend(onenumber.astype(float))
  result = stub.Classify(request, 10.0)
  predicted_num = predicted_num + max(result.result.classifications[0].classes, key=attrgetter('score')).label
print("Predicted Number: {}".format(predicted_num))

