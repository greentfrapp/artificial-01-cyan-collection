"""Main script that created CYAN COLLECTION.
https://www.instagram.com/p/BsPI9GDhARF/

References:
Mordvintsev, et al., "Differentiable Image Parameterizations", Distill, 2018.
https://distill.pub/2018/differentiable-parameterizations/
https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/differentiable-parameterizations/xy2rgb.ipynb
"""

from __future__ import print_function

import numpy as np
import PIL
import tensorflow as tf
from tensorflow.contrib import slim
from lucid.modelzoo import vision_models
from lucid.optvis import objectives
from lucid.optvis import render
from keras.applications import ResNet50
from keras.applications.imagenet_utils \
	import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from matplotlib.colors import to_rgb
from progress.bar import IncrementalBar
from absl import flags, app


FLAGS = flags.FLAGS

# Commands
flags.DEFINE_integer('chosen_class', 949, 'ImageNet class from 0 to 999',
	short_name='class')
flags.DEFINE_string('hue_hex', '#00FFFD', 'Hue to apply onto the image',
	short_name='hue')
flags.DEFINE_integer('steps', 2560, 'Number of optimization steps',
	short_name='steps')
flags.DEFINE_integer('output_size', 1024, 'Size of output image',
	short_name='size')
flags.DEFINE_string('output_path', 'image.jpg', 'Path to save image',
	short_name='path')


def render_vis(model, objective_f, param_f=None, optimizer=None,
	transforms=None, steps=2560, relu_gradient_override=True, output_size=1024,
	output_path='image.jpg'):
	"""Adapted render_vis function from the Lucid library
	https://github.com/tensorflow/lucid/blob/master/lucid/optvis/render.py
	"""

	global _size

	with tf.Graph().as_default() as graph, tf.Session() as sess:

		T = render.make_vis_T(model, objective_f, param_f, optimizer,
			transforms, relu_gradient_override)
		loss, vis_op, t_image = T('loss'), T('vis_op'), T('input')
		tf.global_variables_initializer().run()

		images = []
		bar = IncrementalBar(
			'Creating image...',
			max=steps,
			suffix='%(percent)d%%'
		)
		for i in range(steps):
			sess.run(vis_op, feed_dict={_size: 224})
			bar.next()
		bar.finish()
		print('Saving image as {}.'.format(output_path))
		img = sess.run(t_image, feed_dict={_size: output_size})
		PIL.Image.fromarray((img.reshape(output_size, output_size, 3) * 255)
			.astype(np.uint8)).save(output_path)

def composite_activation(x):
	"""Activation used in CPPN (see below)
	"""
	x = tf.atan(x)
	return tf.concat([x/0.67, (x*x)/0.6], -1)

def cppn(size=None, num_output_channels=3, num_hidden_channels=24,
	num_layers=8, activation_fn=composite_activation, normalize=False):
	"""Function that returns a Tensor output from a CPPN.
	Adapted from CPPN Colab notebook from 
	Mordvintsev, et al., "Differentiable Image Parameterizations", Distill, 2018.
	See References up top.
	"""
	
	global _size
	_size = tf.placeholder(
		shape=None,
		dtype=tf.int32,
	)

	r = 3.0**0.5
	coord_range = tf.linspace(-r, r, _size)
	y, x = tf.meshgrid(coord_range, coord_range, indexing='ij')
	net = tf.expand_dims(tf.stack([x,y], -1), 0)

	with slim.arg_scope([slim.conv2d], kernel_size=1, activation_fn=None):
		colors = tf.constant(
			value=np.array([[[list(to_rgb(FLAGS.hue_hex))]]]),
			dtype=tf.float32,
		)
		for i in range(num_layers):
			in_n = int(net.shape[-1])
			net = slim.conv2d(
				net,
				num_hidden_channels,
				weights_initializer=tf.random_normal_initializer(
					0.0, np.sqrt(1.0/in_n)
				),
			)
			if normalize:
				net = slim.instance_norm(net)
			net = activation_fn(net)
		rgb = slim.conv2d(
			net,
			num_output_channels,
			activation_fn=tf.nn.sigmoid,
			weights_initializer=tf.zeros_initializer()
		)
		rgb = tf.clip_by_value(rgb * colors, 0, 1)

	return rgb

def classify(filename):
	print('\nLoading ResNet50 from Keras...')
	classifier = ResNet50(weights='imagenet')
	image = load_img(filename, target_size=(224, 224))
	image = np.expand_dims(img_to_array(image), axis=0)
	image = preprocess_input(image)
	predictions = decode_predictions(classifier.predict(image))
	print('\n------------------------------')
	print('Predictions')
	print('------------------------------')
	for i, prediction in enumerate(predictions[0]):
		label_id, label_name, confidence = prediction
		print('{}. {}: {:.3f}%'
			.format(i + 1, label_name, confidence * 100))

def main(unused_args):
	model = vision_models.ResnetV1_50_slim()
	model.load_graphdef()
	print('------------------------------')
	print('Loaded Parameters')
	print('------------------------------')
	print('Chosen Class: {}'.format(FLAGS.chosen_class))
	print('Hue         : {}'.format(FLAGS.hue_hex))
	print('No. of Steps: {}'.format(FLAGS.steps))
	print('Output Size : {}'.format(FLAGS.output_size))
	print('Output Path : {}'.format(FLAGS.output_path))
	print('------------------------------')
	render_vis(
		model=model,
		objective_f=objectives.channel(
			'resnet_v1_50/SpatialSqueeze',
			FLAGS.chosen_class,
		),
		param_f=lambda:cppn(),
		optimizer=tf.train.AdamOptimizer(5e-3),
		transforms=[],
		steps=FLAGS.steps,
		output_size=FLAGS.output_size,
		output_path=FLAGS.output_path,
	)
	classify(FLAGS.output_path)


if __name__ == '__main__':
	app.run(main)
