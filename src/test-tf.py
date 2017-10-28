# from keras.models import load_model
# import tensorflow as tf
# from keras import backend as K
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
# # Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.
# # Without explicitly linking the session the weights for the dense layer added below don't get loaded
# # and so the model returns random results which vary with each model you upload because of random seeds.
# K.set_session(sess)
#
# # Use this only for export of the model.
# # This must come before the instantiation of ResNet50
# K._LEARNING_PHASE = tf.constant(0)
# K.set_learning_phase(0)
#
# model = load_model('sensor_model_full.hdf5')
#
# from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
#
# # I want the full prediction tensor out, not classification. This format: {"image": Resnet50model.input} took me a while to track down
# prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"image": model.input}, {"prediction":model.output})
#
# # export_path is a directory in which the model will be created
# builder = saved_model_builder.SavedModelBuilder("models/tf-serving")
# legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#
# # Initialize global variables and the model
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)
#
# # Add the meta_graph and the variables to the builder
# builder.add_meta_graph_and_variables(
#       sess, [tag_constants.SERVING],
#       signature_def_map={
#            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                prediction_signature,
#       },
#       legacy_init_op=legacy_init_op)
# # save the graph
# builder.save()

import tensorflow as tf
from tensorflow.python.platform import gfile
from MultimodalDataset import MultimodalDataset
import numpy as np
from SensorAgent import SensorAgent
from VisionAgent import VisionAgent
sns_x, sns_y = next(MultimodalDataset.flow_from_dir(root='multimodal_dataset/video/splits/train', max_frames_per_video=150,
                                group_size=10,
                                batch_size=1, type="img"))
with tf.Session() as sess:
    model_filename = 'models/tensorflow_model/vision_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input_x, output_x= tf.import_graph_def(graph_def, return_elements=['input_1:0', 'output_node0:0'])
        op = sess.graph.get_operations()
        # for m in op:
        #     print(m.values())
        # assign_ops = [op for op in tf.Graph.get_operations(sess.graph)
        #               if 'assign_ops' in op.name]
        #sess.run(assign_ops)
        print(np.argmax(sess.run([output_x], {input_x: np.squeeze(sns_x)})))

agent = VisionAgent(weights='models/vision_model.hdf5', architecture="inception", tf_input=input_x, tf_output=output_x)
print(agent.predict_from_tf(np.squeeze(sns_x)))
print(agent.predict(np.squeeze(sns_x)))

