VAE Tutorial
============

This repository contains a series of small exercises to walk you, dear reader, through some of the core concepts of Variational Auto-Encoders (VAEs) by having you implement your own miniature VAE library in [Keras](http://github.com/fchollet/keras). Completing this tutorial will give you a better understanding not only of how probabilistic deep-learning libraries like [Edward](https://github.com/blei-lab/edward), [PyRo](https://github.com/uber/pyro), [ZhuSuan](https://github.com/thu-ml/zhusuan/), or others work under the hood, but also of how to write custom extensions in Keras in general.

Pre-requisites
--------------

* Familiarity with core concepts in Keras - make sure you understand what is happening in Keras' [getting started](https://keras.io/#getting-started-30-seconds-to-keras) and [functional API](https://keras.io/getting-started/functional-api-guide/) documentation.
* Familiarity with [keras backend](https://keras.io/backend/) functions like `K.dot()`, `K.mean()`, `K.exp()`, etc., and [how they are used inside keras' `Layer` objects.](https://keras.io/layers/writing-your-own-keras-layers/)
* Familiarity with VAEs - read an introduction such as [this one](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/).

---

Setup
-----

Since the `data/` directory and `visualize.py` are shared by all exercises, start by putting the project's root directory in your `PYTHONPATH` (this will allow other scripts to do `from data import ...` from within other directories). In a terminal, run

    export PYTHONPATH=$PWD:$PYTHONPATH

Next, use [virtualenv](https://virtualenv.pypa.io/en/stable/) to create a python environment and install the necessary dependencies (solutions were written in Python 3.5
 but should work in Python 2.7):

    virtualenv vae-tutorial
    pip install -r requirements.txt

Organization
------------

This tutorial is broken into a series of ordered "exercises," each building on the previous. In the `solutions/` directory, each solution's source files build on the previous ones by only a small "diff". Each exercise contains an "instructions.md" file outlining what you need to implement and giving helpful tips. These are best viewed in-browser - [click here to go to exercise 01](https://github.com/wrongu/vae-tutorial/blob/master/01%20-%20ELBO%20in%20keras/instructions.md).

Each exercise contains `# YOUR CODE HERE` comments across various files indicating parts you must implement.
