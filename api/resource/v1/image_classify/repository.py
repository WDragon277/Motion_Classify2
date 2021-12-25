# -*- coding: utf-8 -*-  
import functools

import matplotlib as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import os

def csv_merge():
    input_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br' # csv파일들이 있는 디렉토리 위치
    output_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br' # 저장하려는 위치

    allFile_list = glob.glob(os.path.join(input_file,'*')) # glob함수로 파일들을 모은다
    print(allFile_list)
    allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
    for file in allFile_list:
        df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
        df.columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다
        datacombine = pd.concat(allData, ignore_index=True)
        MaxMin_datacombined = (datacombine - datacombine.min()) / (datacombine.max() - datacombine.min())
        StEnd_dataremoved = MaxMin_datacombined[100:-100]
    return StEnd_dataremoved

refined_data = csv_merge()
Max_index=refined_data['ax'].count()
brcount = 300
bdcount = 385
cbcount = 200
wscount= 140
split_data =pd.DataFrame()

def data_label(motion):

    if motion == 'bd':
        bins = list(range(0,Max_index,bdcount))
        bins_label = [int(x / bdcount) + 1  for x in bins]
    elif motion == 'br':
        bins = list(range(0, Max_index, brcount))
        bins_label = [int(x / brcount) for x in bins]
        refined_data['try'] = pd.cut(refined_data.index, bins, right=False, labels=bins_label[:-1])
        split_data.append([refined_data.iloc[x:x+300] for x in bins])
        k=0
        for i in split_data:

            i.to_csv(r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br'+ str(k),header=False,index=False)
            k = k+1
        # split_data = pd.DataFrame(split_data)
        # split_data.to_csv(output_file,header=False,index=False)

    elif motion == 'cb':
        bins = list(range(0, Max_index, cbcount))
        bins_label = [int(x / cbcount) for x in bins]
    elif motion == 'ws':
        bins = list(range(0, Max_index, wscount))
        bins_label = [int(x / wscount) for x in bins]

        return

# data normalization
remove row with char(eng), null to 0. 

#df 합치기
datacombine=pd.concat(allData,ignore_index=  False)

#Max-Min
MaxMin_datacombined = (datacombine-datacombine.min()) / (datacombine.max()-datacombine.min())


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

# 6 훈련 과정 시각화 (정확도)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 7 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

