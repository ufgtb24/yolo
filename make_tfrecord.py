import tensorflow as tf
import numpy as np
import cv2
import csv
from datetime import datetime
import os
import sys
import threading


net_scale = 32
grid_w, grid_h = 18, 10
n_classes = 6
iou_th = 0.7
in_w, in_h = grid_w * net_scale, grid_h * net_scale

tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_string('csv_path', 'my_csv.csv',
                           'Output data directory')

tf.app.flags.DEFINE_string('anchors_path', 'anchors.txt',
                           'Output data directory')

tf.app.flags.DEFINE_string('base_name', 'teeth_pic',
                           'Output data directory')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of num_threads.')

tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def read_anchors_file(file_path):
    anchors = []
    with open(file_path, 'r') as file:
        for line in file.read().splitlines():
            anchors.append(map(float, line.split()))
    
    return np.array(anchors)


def iou_wh(r1, r2):
    min_w = min(r1[0], r2[0])
    min_h = min(r1[1], r2[1])
    area_r1 = r1[0] * r1[1]
    area_r2 = r2[0] * r2[1]
    
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    
    return intersect / union


def get_grid_cell(roi, raw_w, raw_h, grid_w, grid_h):
    x_center = roi[0] + roi[2] / 2.0
    y_center = roi[1] + roi[3] / 2.0
    
    grid_x = int(x_center / float(raw_w) * float(grid_w))
    grid_y = int(y_center / float(raw_h) * float(grid_h))
    
    return grid_x, grid_y

# 获得confidence = 1 的anchor. 其余的anchor的confidence都是0. YOLO3 中还有第三类
# 若果要改造成 YOLO3，需要改这个函数
def get_active_anchors(roi, anchors):
    indxs = []
    iou_max, index_max = 0, 0
    for i, a in enumerate(anchors):
        # 5 种 anchor的循环，而不是关于图像中 所有anchor的循环
        # 只求形状大小匹配，位置不要求
        iou = iou_wh(roi[2:], a)
        if iou > iou_th:
            indxs.append(i)
        if iou > iou_max:
            iou_max, index_max = iou, i
    
    if len(indxs) == 0:
        # 实在没有形状匹配的就选相对最好的，总之得选一个，V3就不是这样
        indxs.append(index_max)
    
    return indxs



def read_csv_file(filename):
    filenames = []
    rois = []
    with open(filename) as csvfile:
        i = ['filename', 'rois', 'classes']
        csvdata = csv.DictReader(csvfile)
        for row in csvdata:
            filenames.append(row['filename'])
            rois.append(row['rois'])
    
    return filenames, rois


def roi2label(roi, anchor, raw_w, raw_h, grid_w, grid_h):
    x_center = roi[0] + roi[2] / 2.0
    y_center = roi[1] + roi[3] / 2.0
    
    grid_x = x_center / float(raw_w) * float(grid_w)
    grid_y = y_center / float(raw_h) * float(grid_h)
    
    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)
    
    roi_w_scale = roi[2] / anchor[0]
    roi_h_scale = roi[3] / anchor[1]
    
    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]
    
    return label


def onehot(idx, num):
    ret = np.zeros([num], dtype=np.float32)
    ret[idx] = 1.0
    
    return ret


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def make_example(filename, rois, anchors):
    n_anchors = np.shape(anchors)[0]
    
    # [roi_num,4]  eval 用来计算字符串表达式
    rois = np.array(eval(rois), dtype=np.float32)
    
    img = cv2.imread(filename)
    raw_h = np.shape(img)[0]
    raw_w = np.shape(img)[1]
    img = cv2.resize(img, (in_w, in_h))
    
    label = np.zeros([grid_h, grid_w, n_anchors, 5], dtype=np.float32)
    
    for roi in rois:
        # roi [4,]   cls [1,]
        # IOU larger than threshold, or the highest
        active_indxs = get_active_anchors(roi, anchors)
        grid_x, grid_y = get_grid_cell(roi, raw_w, raw_h, grid_w, grid_h)
        
        for active_indx in active_indxs:
            anchor_label = roi2label(roi, anchors[active_indx], raw_w, raw_h, grid_w, grid_h)
            # label 的格式是 [grid_y, grid_x, n_anchors*5]
            label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [1.0]))
    
    image_raw = img.tostring()
    label_raw = label.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))}))
    return example


def _process_image_files_batch(thread_index, ranges, name, csv_filenames, csv_rois,
                               num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.
  
    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    # 对 thread range 进行进一步的拆分，每个thread 负责多个 shard
    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    anchors = read_anchors_file(FLAGS.anchors_path)
    
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = csv_filenames[i]
            rois = csv_rois[i]
            
            example = make_example(filename, rois, anchors)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        
        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files():
    """Process and save list of images as TFRecord of Example protos.
    """
    name = FLAGS.base_name
    csv_filenames, csv_rois = read_csv_file(FLAGS.csv_path)
    
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(csv_filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    # ranges records the index range in imageset for particular thread to deal with
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    
    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()
    
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    
    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, csv_filenames, csv_rois, FLAGS.train_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    
    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(csv_filenames)))
    sys.stdout.flush()

if __name__ == "__main__":
    
    _process_image_files()
