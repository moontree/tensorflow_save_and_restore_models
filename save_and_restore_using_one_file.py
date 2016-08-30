'''
author : zhang chao
date : 2016-08-26

content:
    use tf.train.Saver() to save variables and graph
    use freeze_graph to transform two files(.meta of graph, .ckpt of weights) into onefiles
    use tf.GraphDef().ParseFromString() to parse protobuffer
    use import_graph_def to restore graph and weights
    use sess.graph.get_tensor_by_name() to get the input and output
    test
'''


import numpy as np
import tensorflow as tf
import freeze_graph

input_graph_path = 'graphs/test1.pb'
input_saver_def_path = ""
input_binary = True
input_checkpoint_path = 'test.ckpt'
output_node_names = "output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = 'test.pb'
clear_devices = False

def common_save_net_work():
    x = tf.placeholder('float32',[None,1], name = 'data')
    y_ = tf.placeholder('float32',[None,1])
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = 'weight')
    b = tf.Variable(tf.zeros([1]), name = 'bias')
    y = tf.add(W * x , b, name = 'output')

    loss = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    test_x = np.array([[0.],[1.],[2.]])
    test_y = np.array([[2.],[4.],[6.]])
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    for i in range(40):
        _, los = sess.run([train_step, loss], feed_dict = {x : test_x, y_ : test_y})
        print "step = %d, loss = %f" % (i, los)
    tf.train.write_graph(sess.graph.as_graph_def(), "graphs/", "test1.pb", False)
    print 'save finish'


# use freeze_graph to save graph and weights into one file, usually .pb
def save_to_one_file():
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")

#import and test
def restore_network_and_test():
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            output_node = sess.graph.get_tensor_by_name("output:0")
            data = sess.graph.get_tensor_by_name("data:0")
            print 'predict use saved model'
            xx = [[3.5],[4],[6]]
            print 'input is ', xx
            print 'output is '
            output = sess.run(output_node, feed_dict={ data : xx})
            print output

def restore_and_finetune():
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            output_node = sess.graph.get_tensor_by_name("output:0")
            data = sess.graph.get_tensor_by_name("data:0")
            print 'predict use saved model'
            xx = [[3.5], [4], [6]]
            print 'input is ', xx
            print 'output is '
            output = sess.run(output_node, feed_dict={data: xx})
            print output

            print 'start finetune:'
            print output_node
            output_node = tf.reshape(output_node,[-1,1])
            print output_node.get_shape()
            W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
            b = tf.Variable(tf.zeros([1]))
            y = tf.add(W * output_node, b)

            y_ = tf.placeholder(tf.float32, [None, 1])
            loss = tf.reduce_sum(tf.square(y - y_))
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            test_x = np.array([[0.], [1.], [2.]])
            test_y = np.array([[1.], [4.], [7.]])
            sess.run(tf.initialize_all_variables())
            for i in range(601):
                _, los = sess.run([train_step, loss], feed_dict={data: test_x, y_: test_y})
                if i % 30 == 0:
                    print "step = %d, loss = %f" % (i, los)
            print sess.run(y, feed_dict={data: [[5], [6], [7]]})

if __name__ == '__main__':
    #common_save_net_work()
    #save_to_one_file()
    #restore_network_and_test()
    restore_and_finetune()