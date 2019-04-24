# Eage Execution Basics

## Overview

This introduction will cover:

- Importing required packages

- Creating and using Tensor

- Using GPU

- Datasests

## Import TensorFlow

To get started, import the TensorFlow and enable eager execution.

```python
import tensorflow as tf

tf.enable_eager_execution()
```

## Tensors

### NumPy Compatibility

Conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:

- TensorFlow operations automatically convert NumPy ndarrays to Tensors

- NumPy operations automatically conver TensorFlow Tensors to NumPy ndarrays

Tensors can be explicitly converted to NumPy ndarrays by invoking the .numpy() method on them.

```python
import numpy as np

ndarray = np.ones([3, 3])

# TensorFlow operations convert numpy arrays to Tensors automatically
tensor = tf.multiply(ndarra, 42)
print(tensor)

# NumPy operations convert Tensors to NumPy ndarrays automatically
print(np.add(tensor, 1)

# The .numpy() method explicity converts a Tensor to a NumPy ndarray
print(tensor.numpy())
```

## Using GPU

### Explicit Device Placement

The term `placement` in TensorFlow refers to how individual operations are assigned (placed on) a device for execution. TensorFlow operations can be explicitly placed on specific devices using the `tf.device` context manager. For example:

```python
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)
        
    result = time.time() - start
    
    print("10 loops: {:0.2f}ms".format(1000 * result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
```

## Datasets

- Creating a `Dataset`

- Iteration over a `Dataset` with eager execution enabled

### Create a source Dataset

```python
ds_tensors = tf.data.Datasests.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, "w") as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)
```

### Apply Transformations

```python
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)
```

### Iterate

## Reference

- https://www.tensorflow.org/tutorials/eager/eager_basics
