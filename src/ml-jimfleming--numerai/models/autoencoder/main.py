from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time
import random
random.seed(67)

import numpy as np
np.random.seed(67)

import pandas as pd

import tensorflow as tf
tf.set_random_seed(67)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from model import Model

import os

from PIL import Image
from tensorflow.contrib.tensorboard.plugins import projector


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30, "")
tf.app.flags.DEFINE_integer('batch_size', 128, "")
tf.app.flags.DEFINE_boolean('denoise', True, "")

if FLAGS.denoise:
    print('Denoising!')
else:
    print('NOT denoising!')

def main(_):
    df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
    df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
    df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

    feature_cols = list(df_train.columns)[:-1]

    X_train = df_train[feature_cols].values
    X_valid = df_valid[feature_cols].values
    X_test = df_test[feature_cols].values

    Y_train = df_train['target'].values
    dimensions = (5, 5)
    single = 5
    sprites = [None] * 2
    sprites[0] = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    sprites[1] = [
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1]
    ]
    sprites[0] = Image.fromarray(np.uint8(sprites[0]) * 0xFF)
    sprites[1] = Image.fromarray(np.uint8(sprites[1]) * 0xFF)
    count = X_train.shape[0]
    size = int(math.ceil(math.sqrt(count)))
    image = Image.new('1', (size * single, size * single))
    logdir = os.path.join(os.getenv('STORING'), 'logs', 'autoencoder_{}'.format(int(time.time())))
    os.makedirs(logdir, exist_ok=True)
    handle = open(os.path.join(logdir, 'metadata.tsv'), 'wb')
    for i in range(count):
        location = ((i % size) * single, (i // size) * single)
        label = int(Y_train[i])
        image.paste(sprites[label], location)
        handle.write(b'1\n' if label == 1 else b'0\n')
    handle.close()
    image.save(os.path.join(logdir, 'sprites.png'))

    num_features = len(feature_cols)
    features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')

    with tf.name_scope('training'):
        with tf.variable_scope('autoencoder'):
            train_model = Model(num_features, features, denoise=FLAGS.denoise, is_training=True)

    with tf.name_scope('evaluation'):
        with tf.variable_scope('autoencoder', reuse=True):
            test_model = Model(num_features, features, denoise=FLAGS.denoise, is_training=False)

    best = None
    wait = 0
    summary_op = tf.summary.merge_all()
    supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
    with supervisor.managed_session() as sess:
        print('Training model with {} parameters...'.format(train_model.num_parameters))
        with tqdm(total=FLAGS.num_epochs) as pbar:
            for epoch in range(FLAGS.num_epochs):
                summary_writer = tf.summary.FileWriter(logdir, sess.graph)

                X_train_epoch = shuffle(X_train)

                losses = []
                _, loss = sess.run([
                    train_model.train_step,
                    train_model.loss,
                ], feed_dict={
                    features: X_train_epoch,
                })
                losses.append(loss)

                loss_train = np.mean(losses)

                loss_valid, summary_str = sess.run([
                    test_model.loss,
                    summary_op,
                ], feed_dict={
                    features: X_valid,
                })
                if best is None or loss_valid < best:
                    best = loss_valid
                    wait = 0
                else:
                    wait += 1
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()
                pbar.set_description('[{}] loss (train): {:.8f}, loss (valid): {:.8f} [best: {:.8f}, wait: {}]' \
                    .format(epoch, loss_train, loss_valid, best, wait))
                pbar.update()

        summary_writer.add_graph(sess.graph)

        loss_valid = sess.run(test_model.loss, feed_dict={
            features: X_valid,
        })
        print('Validation loss: {}'.format(loss_valid))

        z_train = sess.run(test_model.z, feed_dict={ features: X_train })
        z_valid = sess.run(test_model.z, feed_dict={ features: X_valid })
        z_test = sess.run(test_model.z, feed_dict={ features: X_test })

        summary_writer.flush()
        summary_writer.close()

        if FLAGS.denoise:
            np.savez(os.path.join(os.getenv('STORING'), 'denoising.npz'), z_train=z_train, z_valid=z_valid, z_test=z_test)
        else:
            np.savez(os.path.join(os.getenv('STORING'), 'autoencoder.npz'), z_train=z_train, z_valid=z_valid, z_test=z_test)

    tf.reset_default_graph()
    tf.Graph().as_default()
    embedding_variable = tf.Variable(z_train, name='autoencoder_embedding')
    summary_writer = tf.summary.FileWriter(logdir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_variable.name
    embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')
    embedding.sprite.image_path = os.path.join(logdir, 'sprites.png')
    embedding.sprite.single_image_dim.extend(dimensions)
    projector.visualize_embeddings(summary_writer, config)
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(session, os.path.join(logdir, 'model.ckpt'), 0)


if __name__ == "__main__":
    tf.app.run()
