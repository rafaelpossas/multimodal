from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

import argparse


def convert_to_pb(model_file, output_fld='models/production/tb_models/'):

    num_output = 1
    write_graph_def_ascii_flag = True
    prefix_output_node_names_of_final_network = 'output_node'

    base_model_file = "_".join(model_file.split(os.path.sep)[-1].split(".")[:-1])

    pb_file = base_model_file + ".pb"
    ascii_file = base_model_file + ".ascii"

    if not os.path.isdir(output_fld):
        os.makedirs(output_fld,exist_ok=True)

    K.set_learning_phase(0)
    net_model = load_model(model_file)

    pred = [None]*num_output
    pred_node_names = [None]*num_output

    for i in range(num_output):
        pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    print('output nodes names are: ', pred_node_names)

    sess = K.get_session()

    if write_graph_def_ascii_flag:
        f = ascii_file
        tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
        print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))

    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, pb_file, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, pb_file))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model_file", default='models/production/multimodal_sns_0.611429-0.71.hdf5')
    a.add_argument("--output_folder", default='models/production/tb_models')

    args = a.parse_args()

    convert_to_pb(args.model_file, args.output_folder)