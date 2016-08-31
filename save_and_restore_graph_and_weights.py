'''
author : zhang chao
date : 2016-08-26

content:
    use tf.add_to_collection() to add the options/tensors/placeholders .. which you want to use to graph keys
    use tf.train.Saver() to save variables
    use tf.train.import_meta_graph() to restore graph
    use tf.train.Saver() to restore variables
    use tf.get_collection() to get what you added to the graph
    test
'''



import numpy as np
import tensorflow as tf



def train_and_save_network():
    sess = tf.Session()
    # net and data
    x = tf.placeholder('float32', [None, 1], name='data')
    y_ = tf.placeholder('float32', [None, 1])

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='weight')
    b = tf.Variable(tf.zeros([1]), name='bias')
    y = tf.add(W * x, b, name='output')

    # add input/x, output/y to graph collections, so that you can use them conveniently
    # name: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.
    # value: The value to add to the collection.
    tf.add_to_collection("input", x)
    tf.add_to_collection("output", y)

    loss = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    test_x = np.array([[0.], [1.], [2.]])
    test_y = np.array([[2.], [4.], [6.]])
    sess.run(tf.initialize_all_variables())
    for i in range(40):
        _, los = sess.run([train_step, loss], feed_dict={x: test_x, y_: test_y})
        print "step = %d, loss = %f" % (i, los)

    save_variables(sess)

# use tf.train.Saver() to save Variables
def save_variables(sess):
    # Pass the variables as a dict:
    #saver = tf.train.Saver({'weight': W, 'bias': b})

    # Or pass them as a list.
    #saver = tf.train.Saver([W, b])

    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    #saver = tf.train.Saver({v.op.name: v for v in [W, b]})

    #Pass nothing to save all variables
    saver = tf.train.Saver()
    # save weights without graph
    # saver.save(sess, 'weights/test.ckpt', write_meta_graph = False)
    # save weights and graph
    saver.save(sess, 'weights/test.ckpt')
    #saver.save(sess, 'my-model', global_step=0) # filename: 'my-model-0'
    print 'save finish'

def restore_from_meta_and_ckpt():
    sess = tf.Session()
    #import saved graph from meta files
    saver = tf.train.import_meta_graph('weights/test.ckpt.meta')
    #restore saved weights
    saver.restore(sess, 'weights/test.ckpt')
    #get the variables/tensors/options added to collection
    x = tf.get_collection("input")[0]
    y = tf.get_collection("output")[0]

    xx = [[4],[6],[7]]
    print 'test saved network:'
    print 'input is : ' , xx
    print 'output is : '
    print sess.run(y , feed_dict={ x: xx})

def restore_and_retrain():
    sess = tf.Session()
    # import saved graph from meta files
    saver = tf.train.import_meta_graph('weights/test.ckpt.meta')
    # restore saved weights
    saver.restore(sess, 'weights/test.ckpt')
    # get the variables/tensors/options added to collection
    x = tf.get_collection("input")[0]
    y = tf.get_collection("output")[0]

    xx = [[4], [6], [7]]
    print 'test saved network:'
    print 'input is : ', xx
    print 'output is : '
    print sess.run(y, feed_dict={x: xx})

    print 'start retrain:'
    y_ = tf.placeholder(tf.float32,[None,1])
    loss = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    test_x = np.array([[0.], [1.], [2.]])
    test_y = np.array([[1.], [4.], [7.]])
    sess.run(tf.initialize_all_variables())
    for i in range(40):
        _, los = sess.run([train_step, loss], feed_dict={x: test_x, y_: test_y})
        print "step = %d, loss = %f" % (i, los)
    print sess.run(y, feed_dict={x: [[5],[6],[7]]})

def restore_and_finetune():
    sess = tf.Session()
    # import saved graph from meta files
    saver = tf.train.import_meta_graph('weights/test.ckpt.meta')

    # get the variables/tensors/options added to collection
    x = tf.get_collection("input")[0]
    y = tf.get_collection("output")[0]
    #tf.stop_gradient(y)
    print 'start finetune:'
    Ww = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    bb = tf.Variable(tf.zeros([1]))
    yy = tf.add(Ww * y, bb)

    y_ = tf.placeholder(tf.float32, [None, 1])
    loss = tf.reduce_sum(tf.square(yy - y_))
    #note that var_list is the Variables you will optimize
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss,var_list=[Ww,bb])

    #do not use initialize_all_variables if you want to keep previous weight unchanged
    #sess.run(tf.initialize_variables([Ww,bb]))

    #notice the order of initialize_all_variables and restore weights from checkpoint
    sess.run(tf.initialize_all_variables())
    # restore saved weights
    saver.restore(sess, 'weights/test.ckpt')

    xx = [[4], [6], [7]]
    print 'test saved network:'
    print 'input is : ', xx
    print 'output is : '
    print sess.run(y, feed_dict={x: xx})
    test_x = np.array([[0.], [1.], [2.]])
    test_y = np.array([[1.], [4.], [7.]])
    for i in range(300):
        _, los = sess.run([train_step, loss], feed_dict={x: test_x, y_: test_y})
        if i % 30 == 0:
            print "step = %d, loss = %f" % (i, los)
    print sess.run(y, feed_dict = {x: [[4],[6],[7]]})
    print sess.run(yy, feed_dict = {x: [[4],[5],[6]]})


if __name__ == '__main__' :
    #train_and_save_network()
    #restore_from_meta_and_ckpt()
    restore_and_finetune()
