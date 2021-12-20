# -*- coding: utf-8 -*-  
""" 
* file description 
    - Copyright ⓒ 2021 KCNET, All rights reserved.
    - fileName : ``repository.py``
    - author : ``이서용 (Lee Seo Yong)``
    - date : ``2021-12-16 오전 12:08``
    - comment : `` ``
       
* revision history 
    - 2021-12-16    Lee Seo Yong    최초 작성
"""
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import os

def csv_merge():
    input_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br' # csv파일들이 있는 디렉토리 위치
    output_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\total_result.csv' # 병합하고 저장하려는 파일명

    allFile_list = glob.glob(os.path.join(input_file,'*')) # glob함수로 파일들을 모은다
    print(allFile_list)
    allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
    for file in allFile_list:
        df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
        allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

    scaler=MinMaxScaler()
    scaler.fit(allData)
    
    minmax_df = scaler.transform(allData)
    # dataCombined <- 최대-최소 정규화 하자

    dataCombined.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인데스를 빼려면 False로 설정
    return






TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

LABEL_COLUMN = 'survived'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True,
      **kwargs)
  return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)




SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path,
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)

example_batch, labels_batch = next(iter(temp_dataset))

def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
  print(features.numpy())
  print()
  print(labels.numpy())
show_batch(raw_train_data)
example_batch, labels_batch = next(iter(temp_dataset))

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
example_batch, labels_batch = next(iter(packed_train_data))

import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column

example_batch['numeric']

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()
