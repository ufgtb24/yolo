import os
import tensorflow as tf

# import numpy as np
# import cv2

SCALE = 32
GRID_W, GRID_H = 18, 10
N_CLASSES = 6
N_ANCHORS = 4
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 3

LAMBDA_COORD = 5.0
LAMBDA_NO_OBJ = 0.5
BATCH_SIZE = 24
LEARNING_RATE = 0.0001
NUM_ITERS = 100000
TFRECORD_PATH = './data.tfrecord'
MODEL_PATH, SAVE_INTERVAL = './model', 10000


# ---------------------------------------------------------------------------------
# -------------------------------- tfrecord reader --------------------------------
# ---------------------------------------------------------------------------------


def preprocess(image_raw):

    images = tf.cast(tf.reshape(image_raw, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), tf.float32)
    images = (images / 255.0-0.5)*2.0
    # images = tf.image.random_flip_left_right(images)
    return  images
    
    


def read_example(filename, batch_size):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    # 从内存队列中读取一张图, one example , and update current position in 内存队列。
    # 内存队列的内容是当前 tfrecord文件的所有examples,如果内存队列被读空，则从文件队列中弹出一个 tfrecord 文件并将其内所有examples放入内存队列
    _, serialized_example = reader.read(filename_queue)
    
    min_queue_examples = 500
    
    batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size, min_after_dequeue=min_queue_examples,
                                   num_threads=2)
    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.string)})


    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    
    label_raw = tf.decode_raw(parsed_example['label'], tf.float32)
    label = tf.reshape(label_raw, [batch_size, GRID_H, GRID_W, N_ANCHORS, 6])
    
    image = preprocess(image_raw)
    return image, label


# ---------------------------------------------------------------------------------
# --------------------------------- net structure ---------------------------------
# ---------------------------------------------------------------------------------

def lrelu(x, leak):
    return tf.maximum(x, leak * x, name='relu')


def maxpool_layer(x, size, stride, name):
    with tf.name_scope(name):
        x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')
    
    return x


def conv_layer(x, kernel, depth, train_logical, name):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             bias_initializer=tf.zeros_initializer())
        x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.99, epsilon=0.001, center=True,
                                          scale=True)
    
    return x


def passthrough_layer(a, b, kernel, depth, size, train_logical, name):
    b = conv_layer(b, kernel, depth, train_logical, name)
    # space_to_depth https: // www.w3cschool.cn / tensorflow_python / tensorflow_python - rkfq2kf9.html
    # 不损失数据量的“下采样”，将size x size x 1 大小的数据块转换成 1 x 1 x (size*size) 的深度块
    b = tf.space_to_depth(b, size)
    y = tf.concat([a, b], axis=3)
    
    return y


def yolo_net(x, train_logical):
    x = conv_layer(x, (3, 3), 32, train_logical, 'conv1')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool1')
    x = conv_layer(x, (3, 3), 64, train_logical, 'conv2')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool2')
    
    x = conv_layer(x, (3, 3), 128, train_logical, 'conv3')
    x = conv_layer(x, (1, 1), 64, train_logical, 'conv4')
    x = conv_layer(x, (3, 3), 128, train_logical, 'conv5')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool5')
    
    x = conv_layer(x, (3, 3), 256, train_logical, 'conv6')
    x = conv_layer(x, (1, 1), 128, train_logical, 'conv7')
    x = conv_layer(x, (3, 3), 256, train_logical, 'conv8')
    x = maxpool_layer(x, (2, 2), (2, 2), 'maxpool8')
    
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv9')
    x = conv_layer(x, (1, 1), 256, train_logical, 'conv10')
    x = conv_layer(x, (3, 3), 512, train_logical, 'conv11')
    x = conv_layer(x, (1, 1), 256, train_logical, 'conv12')
    passthrough = conv_layer(x, (3, 3), 512, train_logical, 'conv13')
    x = maxpool_layer(passthrough, (2, 2), (2, 2), 'maxpool13')
    
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv14')
    x = conv_layer(x, (1, 1), 512, train_logical, 'conv15')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv16')
    x = conv_layer(x, (1, 1), 512, train_logical, 'conv17')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv18')
    
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv19')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv20')
    x = passthrough_layer(x, passthrough, (3, 3), 64, 2, train_logical, 'conv21')
    x = conv_layer(x, (3, 3), 1024, train_logical, 'conv22')
    x = conv_layer(x, (1, 1), N_ANCHORS * (N_CLASSES + 5), train_logical, 'conv23')
    
    y = tf.reshape(x, shape=(-1, GRID_H, GRID_W, N_ANCHORS, N_CLASSES + 5), name='y')
    
    return y


def slice_tensor(x, start, end=None):
    # slice fragment in last dimension
    if end < 0:
        y = x[..., start:]
    
    else:
        if end is None:
            end = start
        y = x[..., start:end + 1]
    
    return y


def yolo_loss(pred, label, lambda_coord, lambda_no_obj):
    # dim 5 of label is the confidence score  [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale, cls, conf ]
    mask = slice_tensor(label, 5)
    # [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale, cls]
    label = slice_tensor(label, 0, 4)
    
    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)), tf.bool)
    
    with tf.name_scope('mask'):
        # boolean_mask results in tensor with dim N-K+1
        # maintain the last (N-K) shape, flattening first K dims where mask is True into 1 dim
        # dim(label)=5 dim(mask)=4 so 5-4+1=2
        # [num_of_anchor_consist_obj,5]
        masked_label = tf.boolean_mask(label, mask)
        # [num_of_anchor_consist_obj,25]
        masked_pred = tf.boolean_mask(pred, mask)
        neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))
    
    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
        masked_pred_wh = tf.exp(slice_tensor(masked_pred, 2, 3))
        masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 4))
        masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 4))
        masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 5, -1))
    
    with tf.name_scope('lab'):
        masked_label_xy = slice_tensor(masked_label, 0, 1)
        masked_label_wh = slice_tensor(masked_label, 2, 3)
        # [-1,1] cls
        masked_label_c = slice_tensor(masked_label, 4)
        masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=N_CLASSES),
                                        shape=(-1, N_CLASSES))
    
    with tf.name_scope('merge'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(tf.square(masked_pred_xy - masked_label_xy))
        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(tf.square(masked_pred_wh - masked_label_wh))
        with tf.name_scope('loss_obj'):
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1))
        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))
        with tf.name_scope('loss_class'):
            loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
        
        loss = lambda_coord * (loss_xy + loss_wh) + loss_obj + lambda_no_obj * loss_no_obj + loss_c
    
    return loss


# ---------------------------------------------------------------------------------
# ----------------------------------- training ------------------------------------
# ---------------------------------------------------------------------------------

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train():
    with tf.name_scope('batch'):
        batch_image, batch_label = read_example(TFRECORD_PATH, BATCH_SIZE)
    
    image = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32,
                           name='image_placeholder')
    label = tf.placeholder(shape=[None, GRID_H, GRID_W, N_ANCHORS, 6], dtype=tf.float32, name='label_palceholder')
    
    train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')
    
    with tf.variable_scope('net'):
        y = yolo_net(image, train_flag)
    with tf.name_scope('loss'):
        loss = yolo_loss(y, label, LAMBDA_COORD, LAMBDA_NO_OBJ)
    
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    saver = tf.train.Saver()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(NUM_ITERS):
        
        image_data, label_data = sess.run([batch_image, batch_label])
        
        _, loss_data, data = sess.run([train_step, loss, y],
                                      feed_dict={train_flag: True, image: image_data, label: label_data})
        
        print('iter: %i, loss: %f' % (i, loss_data))
        
        if (i + 1) % SAVE_INTERVAL == 0:
            make_dir(MODEL_PATH)
            saver.save(sess, os.path.join(MODEL_PATH, 'yolo'), global_step=i + 1)
    
    saver.save(sess, os.path.join(MODEL_PATH, 'yolo'), global_step=i + 1)


if __name__ == "__main__":
    train()
