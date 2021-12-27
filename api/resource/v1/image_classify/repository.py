# -*- coding: utf-8 -*-  
import functools

import matplotlib as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import os

#common에 넣을 변수들.
input_bd_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_bd\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_bd_val_file= r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_bd\CSV'
input_bd_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_bd\raw_CSV'
output_bd_train_file= r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_bd\CSV'
input_br_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_br_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\CSV'
input_br_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_br\raw_CSV'
outpu_br_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_br\CSV'


output_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br'  # 저장하려는 위치


def csv_merge():

    allFile_list = glob.glob(os.path.join(input_bd_train_file,'*')) # glob함수로 파일들을 모은다
    print(allFile_list)
    allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
    for file in allFile_list:
        df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
        df.columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다
        datacombine = pd.concat(allData, ignore_index=True)
        MaxMin_datacombined = (datacombine - datacombine.min()) / (datacombine.max() - datacombine.min())
        StEnd_dataremoved = MaxMin_datacombined[60:-60] # 테스트 준비시간인 초기/말기 3초 데이터 제거
    return StEnd_dataremoved

refined_data = csv_merge()
Max_index=refined_data['ax'].count()
brcount = 300
bdcount = 385
cbcount = 200
wscount= 140
split_data =[]

def data_label(motion):

    if motion == 'bd': #버드독 데이터 csv 저장
        bins = list(range(0,Max_index,bdcount))
        bins_label = [int(x / bdcount) + 1  for x in bins]
        refined_data['try'] = pd.cut(refined_data.index, bins, right=False, labels=bins_label[:-1])
        split_data.append([refined_data.iloc[x:x+300] for x in bins])
        k=0
        for i in split_data[0]:

            i.to_csv(r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_bd\CSV\rf_bd'+ str(k)+ '.csv',header=False,index=False)
            k = k+1

    elif motion == 'br': #브릿지 데이터 csv 저장
        bins = list(range(0, Max_index, brcount))
        bins_label = [int(x / brcount) for x in bins]
        refined_data['try'] = pd.cut(refined_data.index, bins, right=False, labels=bins_label[:-1])
        split_data.append([refined_data.iloc[x:x+300] for x in bins])
        k=0
        for i in split_data[0]:

            i.to_csv(r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\CSV\rf_br'+ str(k)+ '.csv',header=False,index=False)
            k = k+1


    elif motion == 'cb':#코브라 데이터 csv 저장
        bins = list(range(0, Max_index, cbcount))
        bins_label = [int(x / cbcount) for x in bins]
        refined_data['try'] = pd.cut(refined_data.index, bins, right=False, labels=bins_label[:-1])
        split_data.append([refined_data.iloc[x:x+300] for x in bins])
        k=0
        for i in split_data[0]:

            i.to_csv(r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_cb\rf_cb'+ str(k)+ '.csv',header=False,index=False)
            k = k+1

    elif motion == 'ws':#허리펴기 데이터 csv 저장
        bins = list(range(0, Max_index, wscount))
        bins_label = [int(x / wscount) for x in bins]
        refined_data['try'] = pd.cut(refined_data.index, bins, right=False, labels=bins_label[:-1])
        split_data.append([refined_data.iloc[x:x+300] for x in bins])
        k=0
        for i in split_data[0]:

            i.to_csv(r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_ws\rf_ws'+ str(k)+ '.csv',header=False,index=False)
            k = k+1
        return

# train 데이터셋 경로 지정
train_bd_dir = './tmp/horse-or-human/horses'
train_br_dir = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\CSV'
# bd 파일 이름 리스트
train_horse_names = os.listdir(train_bd_dir)
print(train_bd_dir[:10])

# br 파일 이름 리스트
train_human_names = os.listdir(train_br_dir)
print(train_br_dir[:10])

# horses/humans 총 이미지 파일 개수
print('total training horse images:', len(os.listdir(train_bd_dir)))
print('total training human images:', len(os.listdir(train_br_dir)))

# validation 데이터셋 경로 지정

validation_bd_dir =
validation_br_dir =

validation_cats_dir = os.path.join(validation_bd_dir, 'cats')
validation_dogs_dir = os.path.join(validation_br_dir, 'dogs')







#
# data normalization

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

