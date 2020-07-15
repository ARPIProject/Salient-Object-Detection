import os

import numpy as np
import tensorflow as tf
from PIL import Image

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])


def main():
	input_folder = "./input"
	output_folder = "./output"

	with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
			gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.0))) as sess:
		tf.compat.v1.train.import_meta_graph('./inference/meta_graph/my-model.meta').restore(
			sess, tf.train.latest_checkpoint('./inference/salience_model'))

		for filename in [s for s in os.listdir(input_folder) if not s == '.gitkeep']:
			file_path = os.path.join(input_folder, filename)
			img = np.asarray(Image.open(file_path).resize((320, 320), resample=Image.NEAREST).convert('RGB'))
			img = img - g_mean
			img = img.reshape((1, 320, 320, 3))
			feed_dict = {tf.compat.v1.get_collection('image_batch')[0]: img}
			detection = sess.run(tf.compat.v1.get_collection('mask')[0], feed_dict=feed_dict)
			Image.fromarray(detection[0].reshape(320, 320) * 255).convert('RGB').save(os.path.join(output_folder, filename))


if __name__ == '__main__':
	main()
