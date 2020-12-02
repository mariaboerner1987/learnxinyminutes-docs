---
category: tool
tool: Flower Framework
language: python
contributors:
    - ["Dr. Maria Börner", "https://github.com/mariaboerner1987"]
    
---
# Flower- A Friendly Federated Learning Framework

This is a tutorial to run a Keras machine learning task federated with the Flower framework. It is suggested to set up the code within a virtual envriorment as pyenv/pyenv-virtualenv. Flower is a agnostic federated learning framework that sits on top of your existing machine learning code. It is compatible with different programming languages, machine learning frameworks and computing systems.
Setting up a federated machine learning workload is possible within 20 lines code by using Flower and the following example.
The shown examples is based on python and requires basic python knowledge as well as a basic machine learning knowledge.

## Why Flower?

The concept of federated learning was inventend by [Google](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html) to train AI models on clients, aggregate all client models and create a global model that is distributed back to the client. Flower provides the required infrastructure to connect several clients with their existing machine learning models to one server aggregating these models.

## Flower Quickstarter

Before you start using Flower install the Flower package. It is recommended to install it with pip or poetry:

```
pip install flwr
```

or

```
poetry add flwr
```

```python

"""
First, the clients (client.py) are  created with the following example. All clients are the same the given example. 
Since this example is based on Keras the tensorflow package is required to run the machine learning training.
If you want to use a different machine learning example you can also import pytorch, scikit, ...
"""
import tensorflow as tf

"""
Import the flower package to run your machine learning workload federated.
"""

import flwr as fl

"""
First, load the dataset that you want to train with

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.XXX.load_data()

Keras.datasets provides already a set of samples to perform the training (x_train, y_train) and another set of samples to test the model parameter (x_test, y_test).

This tutorial concentrates on images with a multi-class classification. Different muti-class image datasets are available within Keras:

- MNIST (hand written digits (0-9) in greyscale)
- Fashion MNIST (greyscale images for 10 different fashion categories as dress, shirt, etc.)
- CIFAR10 (colored images with 10 categories as bird, automobile, flowers, etc.)

"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

"""
Load a Keras model.
You can choose to load a sequential keras model with

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

or any of the available Keras application model that is given in https://keras.io/api/applications/ as MobileNetV2, ResNet, ....

"""
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)

"""
Compile the model by setting the loss (error) and optimizer with

model.compile(optimizer= " ", loss= " ")

Typical optmizer are:
- adam
- sgd (Gradient Descent)

Typical loss functions for multi-class classification are:
- categrorical crossentropy
- sparse categorial crossentropy
- kullback leibler divergence
"""
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


"""
Definition of Flower client

Flower needs 2 main function (evaluate and fit) and a helper function (get_weights).

The functionality of the Flower client is the following:
1. train the local Keras model on the client
2. update the local weights of the Keras model
3. take the weights and evaluate the Keras model by measuring the accuracy and loss
4. send the weights to the Flower server
5. receive the updated weights from the Flower server
6. evaluate the updated weights by measuring the accuracy and loss

"""
class CifarClient(fl.client.keras_client.KerasClient):

"""
get_weights():
The get-weight function receives the model weights.
"""

    def get_weights(self):
        return model.get_weights()

"""
fit():
The fit-function trains the Keras model on each connected client.
It takes weights from the server (model.set_weights), trains the client models (model.fit) and updates the weights on the client (model.get_weights).
"""

    def fit(self, weights, config):
        model.set_weights(weights)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), len(x_train)

"""
evaluate():
The evaluate-function measures the loss and accuracy of the model. It takes the weights (model.set_weights) and measures the accuracy and loss of the trained model (model.evaluate) based on the client test set sample.
"""

    def evaluate(self, weights, config):
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_test, y_test)
        return len(x_test), loss, accuracy

"""
The Flower client is started with

fl.client.start_keras_client()

or

fl.client.start_client()

the a generic example. The fl.client.start_client() function sets up the connection to the server and sends the weights from the client model.
"""

fl.client.start_keras_client(server_address="[::]:8080", client=CifarClient())
```


```python
"""
The Flower server (sever.py) needs also the flwr package. 
"""

import flwr as fl

"""
The Flower server is started with only one line of code:
fl.server.start_server
The server takes the model weights from all clients and waits until all clients send their weight updates. As soon as the server receives all weights it starts to average the weights with the FedAvg (Federated Averaging) algorithm. After running the FedAvg algorithm 3 times the server sends the updated weights back to all connected clients.

"""
fl.server.start_server(config={"num_rounds": 3})
```

The federated machine learning workload is started by starting first the server in one terminal:

```shell
python server.py
```

The clients are started by opening two additional terminals and run in both of them the same client.py:

```shell
python client.py
```

The code shows now the training process. Congratulation, you run your first federated learning workload. 

## References

Flower has more examples available that are explained in the Flower documentation at [flower.dev](https://flower.dev/docs).
