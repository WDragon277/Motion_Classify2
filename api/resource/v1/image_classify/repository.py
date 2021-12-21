# -*- coding: utf-8 -*-  
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

   # allData's type is list. Need to make DataFrame. 
datacombined = pd.concat(allData,axis=0,ignore_index=True)

#Max-Min
MaxMin_datacombined = (datacombine-datacombine.min()) / (datacombine.max()-datacombine.min())

    
    minmax_df = scaler.transform(allData)
    # dataCombined <- 최대-최소 정규화 하자

    dataCombined.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인데스를 빼려면 False로 설정
    return


# Make pipline
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))


# Make model
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

# train model
model = get_compiled_model()
model.fit(train_dataset, epochs=15)



