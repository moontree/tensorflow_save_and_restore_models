'''
author : zhang chao
date : 2016-08-26

content:
    use tf.train.Saver() to save variables
    use tf.train.Saver() to restore variables
    test
'''



import numpy as np
import tensorflow as tf

# net and data
x = tf.placeholder('float32',[None,1], name = 'data')
y_ = tf.placeholder('float32',[None,1])

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name = 'weight')
b = tf.Variable(tf.zeros([1]), name = 'bias')
y = tf.add(W * x , b, name = 'output')

sess = tf.Session()

def train_network():
    loss = tf.reduce_sum(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    test_x = np.array([[0.], [1.], [2.]])
    test_y = np.array([[2.], [4.], [6.]])
    sess.run(tf.initialize_all_variables())
    for i in range(40):
        _, los = sess.run([train_step, loss], feed_dict={x: test_x, y_: test_y})
        print "step = %d, loss = %f" % (i, los)


# use tf.train.Saver() to save Variables
def save_variables():
    train_network()

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

def restore_variables():
    saver = tf.train.Saver()
    saver.restore(sess, 'weights/test.ckpt')
    print 'restore weights from weights/test.ckpt successfully, predict by saved weights:'
    xx = [[4],[5],[6]]
    print 'input is ', xx
    print 'result is '
    #test
    print sess.run(y, feed_dict = { x : xx})

if __name__ == '__main__' :
    save_variables()
    restore_variables()
