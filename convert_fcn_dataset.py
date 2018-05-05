#!/usr/bin/env python3
import logging
import os
import io
import PIL.Image
import hashlib

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

from lxml import etree
import re

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS



def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def recursive_parse_xml_to_dict(xml):
  if len(xml) == 0:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}


# os.environ['CUDA_VISIBLE_DEVICES'] = ''

classe_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    encoded_jpg_io = io.BytesIO(encoded_data)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_data).hexdigest()
    
    
    img_label = cv2.imread(label) 
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()
    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 保证最后随机裁剪的尺寸      
        return None    
    
    xml_path = label.replace('SegmentationClass', 'Annotations').replace('.png', '.xml')
    if not os.path.exists(xml_path):
      raise ValueError('Could not find %s, ignoring example.', xml_path)
    with tf.gfile.GFile(xml_path, 'r') as fid:
      xml_str = fid.read()
    xml_data = recursive_parse_xml_to_dict(etree.fromstring(xml_str))['annotation']  
    
    width = int(xml_data['size']['width'])
    height = int(xml_data['size']['height']) 
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
   
    if 'object' in xml_data:
      for obj in xml_data['object']:
       
        difficult = bool(int(obj['difficult']))       
        difficult_obj.append(int(difficult))
        if 1:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)      
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes.append(classe_list.index(class_name))       
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
 
    
    # Your code here, fill the dict
    feature_dict = {
        'image/height': int64_feature(int(height)),
        'image/width': int64_feature(int(width)),
        'image/filename': bytes_feature(xml_data['filename'].encode('utf8')),
        'image/source_id': bytes_feature(xml_data['filename'].encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_data),
        'image/label': bytes_feature(encoded_label),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
        'image/object/difficult': int64_list_feature(difficult_obj),
        'image/object/truncated': int64_list_feature(truncated),
        'image/object/view': bytes_list_feature(poses),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))  
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    writer = tf.python_io.TFRecordWriter(output_filename)
    for data, label in file_pars:
        if not os.path.exists(data):
            logging.warning('Could not find %s, ignoring example.', data)
            continue
        if not os.path.exists(label):
            logging.warning('Could not find %s, ignoring example.', label)
            continue
               
        try:
            tf_example = dict_to_tf_example(data, label)
            if tf_example:
                writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', label)            

    writer.close()   


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
