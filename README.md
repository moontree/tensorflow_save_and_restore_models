# Tensorflow_Save_And_Restore_Models

## Different ways and usage of save and restore trained models using tensorflow

There are several different ways of dumping a TensorFlow graph into a file and then loading it into another program, this project provides clear examples/information on how they work:

* The checkpoint files (produced e.g. by calling [saver.save()](https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops.html#Saver.save) on a [tf.train.Saver](https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops.html#Saver) object) contain only the weights, and any other variables defined in the same program. To use them in another program, you must re-create the associated graph structure (e.g. by running code to build it again, or calling [tf.import_graph_def()](https://www.tensorflow.org/versions/r0.10/api_docs/python/framework.html#import_graph_def)), which tells TensorFlow what to do with those weights. Note that calling saver.save() also produces a file containing a [MetaGraphDef](https://www.tensorflow.org/versions/r0.10/how_tos/meta_graph/index.html), which contains a graph and details of how to associate the weights from a checkpoint with that graph. See the [tutorial](https://www.tensorflow.org/versions/r0.10/how_tos/meta_graph/index.html) for more details.* 
* Freeze the graph to save the graph and weights together using [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py). A frozen graph can be loaded using [tf.import_graph_def()](https://www.tensorflow.org/versions/r0.10/api_docs/python/framework.html#import_graph_def)(This file may be not found in your tensorflow, you can save it to your project). In this case, the weights are (typically) embedded in the graph, so you don't need to load a separate checkpoint.

## Different Ways to Fine Tune models with Previous Weights Unchanged

* Using Tensor/Variables: Using [tf.initialize_variables(var_list)](https://www.tensorflow.org/versions/r0.10/api_docs/python/state_ops.html#initialize_variables) to initialize variables you need, and use [tf.train.Optimizer.minize(loss,var_list)](https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#Optimizer) to optimize the variables.
* Using Constants: Using freeze_graph to store variables as constants



## Refrences:
* [TensorFlow saving into/loading a graph from a file](http://stackoverflow.com/questions/38947658/tensorflow-saving-into-loading-a-graph-from-a-file)

* [Is there an example on how to generate protobuf files holding trained Tensorflow graphs](http://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow?rq=1)

* [How can I execute a TensorFlow graph from a protobuf in C++?](http://stackoverflow.com/questions/34353160/how-can-i-execute-a-tensorflow-graph-from-a-protobuf-in-c?rq=1)

* [A Tool Developer's Guide to TensorFlow Model Files](https://www.tensorflow.org/versions/r0.9/how_tos/tool_developers/index.html#a-tool-developers-guide-to-tensorflow-model-files)

* [Tensorflow APIs](https://www.tensorflow.org/versions/r0.9/api_docs/index.html)

* [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)

* [tf_restore_graph.py](https://gist.github.com/nikitakit/6ef3b72be67b86cb7868)

