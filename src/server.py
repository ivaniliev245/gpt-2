#!/usr/bin/env python3

import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

import model, sample, encoder

app = Flask(__name__)

# Configuration variables
MODEL_NAME = '124M'
MODELS_DIR = os.path.expanduser(os.path.expandvars('models'))
LENGTH = 100
TEMPERATURE = 1
TOP_K = 40
TOP_P = 1

# Load the model and initialize TensorFlow session
enc = encoder.get_encoder(MODEL_NAME, MODELS_DIR)
hparams = model.default_hparams()
with open(os.path.join(MODELS_DIR, MODEL_NAME, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if LENGTH > hparams.n_ctx:
    raise ValueError(f"Cannot generate text longer than window size: {hparams.n_ctx}")

# TensorFlow session setup
sess = tf.Session(graph=tf.Graph())
with sess.graph.as_default():
    context = tf.placeholder(tf.int32, [1, None])
    output = sample.sample_sequence(
        hparams=hparams, length=LENGTH,
        context=context, batch_size=1,
        temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P
    )
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join(MODELS_DIR, MODEL_NAME))
    saver.restore(sess, ckpt)

@app.route('/generate', methods=['POST'])

def generate():
    """Generate text based on the provided prompt."""
    data = request.get_json()
    print(f"Received data: {data}")  # Log the received data for debugging

    raw_text = data.get("prompt", "").strip()
    if not raw_text:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    context_tokens = enc.encode(raw_text)
    out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
    text = enc.decode(out[0])
    return jsonify({"response": text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
