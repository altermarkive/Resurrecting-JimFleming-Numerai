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

from tqdm import tqdm
from model import Model

from sklearn.utils import shuffle

import os

from PIL import Image
from tensorflow.contrib.tensorboard.plugins import projector


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 30, "")
tf.app.flags.DEFINE_integer('batch_size', 128, "")

def main(_):
    df_train = pd.read_csv(os.getenv('PREPARED_TRAINING'))
    df_valid = pd.read_csv(os.getenv('PREPARED_VALIDATING'))
    df_test = pd.read_csv(os.getenv('PREPARED_TESTING'))

    feature_cols = list(df_train.columns[:-1])
    target_col = df_train.columns[-1]

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_valid = df_valid[feature_cols].values
    y_valid = df_valid[target_col].values

    X_test = df_test[feature_cols].values

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
    logdir = os.path.join(
        os.getenv('STORING'),
        'logs',
        'adversarial_{}'.format(int(time.time())))
    os.makedirs(logdir, exist_ok=True)
    handle = open(os.path.join(logdir, 'metadata.tsv'), 'wb')
    for i in range(count):
        location = ((i % size) * single, (i // size) * single)
        label = int(y_train[i])
        image.paste(sprites[label], location)
        handle.write(b'1\n' if label == 1 else b'0\n')
    handle.close()
    image.save(os.path.join(logdir, 'sprites.png'))

    num_features = len(feature_cols)
    features = tf.placeholder(tf.float32, shape=[None, num_features], name='features')
    targets = tf.placeholder(tf.int32, shape=[None], name='targets')

    with tf.name_scope('training'):
        with tf.variable_scope('adversarial'):
            train_model = Model(num_features, features, targets, is_training=True)

    with tf.name_scope('evaluation'):
        with tf.variable_scope('adversarial', reuse=True):
            test_model = Model(num_features, features, targets, is_training=False)

    summary_op = tf.summary.merge_all()
    supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
    with supervisor.managed_session() as sess:
        print('Training model with {} parameters...'.format(train_model.num_parameters))
        optimize_d, optimize_g = True, True
        with tqdm(total=FLAGS.num_epochs) as pbar:
            for epoch in range(FLAGS.num_epochs):
                summary_writer = tf.summary.FileWriter(logdir, sess.graph)

                X_train_epoch, y_train_epoch = shuffle(X_train, y_train)

                losses_d, losses_g = [], []
                if optimize_d:
                    _, loss_d = sess.run([
                        train_model.train_step_d,
                        train_model.loss_d,
                    ], feed_dict={
                        features: X_train_epoch,
                        targets: y_train_epoch,
                    })
                else:
                    loss_d = sess.run(train_model.loss_d, feed_dict={
                        features: X_train_epoch,
                        targets: y_train_epoch,
                    })

                if optimize_g:
                    _, loss_g = sess.run([
                        train_model.train_step_g,
                        train_model.loss_g,
                    ], feed_dict={
                        features: X_train_epoch,
                        targets: y_train_epoch,
                    })
                else:
                    loss_g = sess.run(train_model.loss_g, feed_dict={
                        features: X_train_epoch,
                        targets: y_train_epoch,
                    })

                losses_d.append(loss_d)
                losses_g.append(loss_g)

                loss_train_d = np.mean(losses_d)
                loss_train_g = np.mean(losses_g)

                summary_str = sess.run(summary_op, feed_dict={
                    features: X_valid,
                    targets: y_valid,
                })

                optimize_d = epoch % 2 == 0
                optimize_g = True

                if not optimize_d and not optimize_g:
                    optimize_d = True
                    optimize_g = True

                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

                pbar.set_description('[{}] loss_train_d ({}): {:.8f}, loss_train_g ({}): {:.8f}'.format(epoch, optimize_d, loss_train_d, optimize_g, loss_train_g))
                pbar.update()

        summary_writer.add_graph(sess.graph)

        loss_valid_d, loss_valid_g, summary_str = sess.run([
            test_model.loss_d,
            test_model.loss_g,
            summary_op,
        ], feed_dict={
            features: X_valid,
            targets: y_valid,
        })
        print('Validation loss (d): {:.8f}, loss (g): {:.8f}'.format(loss_valid_d, loss_valid_g))

        z_train = sess.run(test_model.z, feed_dict={ features: X_train })
        z_valid = sess.run(test_model.z, feed_dict={ features: X_valid })
        z_test = sess.run(test_model.z, feed_dict={ features: X_test })

        summary_writer.flush()
        summary_writer.close()

        np.savez(os.path.join(os.getenv('STORING'), 'adversarial.npz'), z_train=z_train, z_valid=z_valid, z_test=z_test)

    tf.reset_default_graph()
    tf.Graph().as_default()
    embedding_variable = tf.Variable(z_train, name='adversarial_embedding')
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
