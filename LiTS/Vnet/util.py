from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf
import numpy as np


def getdice(Y_pred,Y_gt,K=255):
    intersection=2*np.sum(Y_pred[Y_gt==K])
    denominator=np.sum(Y_pred)+np.sum(Y_gt)
    loss=(intersection/denominator)
    return loss
    


def convertMetaModelToPbModel(meta_model, pb_model):
    # Step 1
    # import the model metagraph
    saver = tf.train.import_meta_graph(meta_model + '.meta', clear_devices=True)
    # make that as the default graph
    graph = tf.get_default_graph()
    sess = tf.Session()
    # now restore the variables
    saver.restore(sess, meta_model)
    # Step 2
    # Find the output name
    for op in graph.get_operations():
        print(op.name)
    # Step 3
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session
        sess.graph_def,  # input_graph_def is useful for retrieving the nodes
        ["Placeholder", "output/Sigmoid"])

    # Step 4
    # output folder
    output_fld = './'
    # output pb file name
    output_model_file = 'model.pb'
    # write the graph
    graph_io.write_graph(output_graph_def, pb_model + output_fld, output_model_file, as_text=False)
