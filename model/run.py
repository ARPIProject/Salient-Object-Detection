import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import copy

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])
IMAGE_SIDE = 320

st.title("Salient Object Detection")
image_file = st.sidebar.file_uploader("Choose a PNG or JPEG file.", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
    image = Image.open(image_file).resize((IMAGE_SIDE, IMAGE_SIDE), resample=Image.NEAREST)

    input_image = (np.asarray(image.convert('RGB')) - g_mean).reshape((1, IMAGE_SIDE, IMAGE_SIDE, 3))

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.0))) as sess:
        tf.compat.v1.train.import_meta_graph('meta_graph/my-model.meta').restore(
            sess, tf.train.latest_checkpoint('salience_model'))

        feed_dict = {tf.compat.v1.get_collection('image_batch')[0]: input_image}
        detection = sess.run(tf.compat.v1.get_collection('mask')[0], feed_dict=feed_dict)

    salient_mask = Image.fromarray(detection[0].reshape(IMAGE_SIDE, IMAGE_SIDE) * 255).convert('L')

    masked_image = copy.deepcopy(image)
    masked_image.putalpha(salient_mask)

    st.subheader("Original Image alongside Detected Mask and Final Image Masking")
    st.image([image, salient_mask, masked_image])
else:
    st.header("No image loaded. Load one from the sidebar!")
