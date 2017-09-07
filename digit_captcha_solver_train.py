from __future__ import print_function
import tensorflow as tf
import numpy
import glob2
import ntpath

from image_preprocess import get_arrays_from_img


def get_samples(imfold):
    train_lab = []
    train_s = []
    lst = glob2.glob(imfold+"*.png")
    for ifile in lst:
        fname = ntpath.basename(ifile)
        clables = list(fname.split('.')[0])
        ilabels = list(map(int, clables))
        train_lab = train_lab + ilabels
        train_s = train_s + get_arrays_from_img(ifile)
    return [numpy.asarray(train_lab),numpy.asarray(train_s)]


# Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 64
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 70 # data input (img shape: 7 x 10)
num_classes = 10 # total classes (0-9 digits)

feature_columns = [tf.feature_column.numeric_column("x", shape=[num_input])]
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[n_hidden_1, n_hidden_2],
                                          n_classes=num_classes,
                                          model_dir="/tmp/model_dir")

trainfold = "train/"
train_labels,train_set = get_samples(trainfold)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": numpy.array(train_set)},
      y=numpy.array(train_labels),
      num_epochs=None,
      shuffle=True)
print("Training Model")
classifier.train(input_fn=train_input_fn, steps=num_steps)

# Evaluate accuracy.
testfold = "test/"
test_labels,test_set = get_samples(testfold)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": numpy.array(test_set)},
  y=numpy.array(test_labels),
  num_epochs=1,
  shuffle=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("Test Accuracy: {0:f}".format(accuracy_score))

#Predict new sample
new_samples = numpy.array(get_arrays_from_img('predict_sample.png'))
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": new_samples},
  num_epochs=1,
  shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]
predicted_num = list(map(lambda x: str(int(x[0])),predicted_classes))
final_num = ''.join(predicted_num)
print("Predicted Number : {}".format(final_num))


#Saving model to be used in captcha solver client
feature_spec = {'x': tf.FixedLenFeature(shape=[70], dtype=tf.float32) }
def serving_input_receiver_fn():
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

classifier.export_savedmodel("/tmp/savedmodel", serving_input_receiver_fn)
print("Model exported")