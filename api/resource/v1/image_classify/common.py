import os


test_data_dir = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\11. datagenerator\test'
train_data_dir = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\11. datagenerator\train'


input_bd_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_bd\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_bd_val_file= os.path.join(test_data_dir,'rf_bd')
input_bd_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_bd\raw_CSV'
output_bd_train_file= os.path.join(train_data_dir,'rf_bd')

input_br_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_br\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_br_val_file = os.path.join(test_data_dir,'rf_br')
input_br_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_br\raw_CSV'
outpu_br_train_file = os.path.join(train_data_dir,'rf_br')

input_cb_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_cb\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_cb_val_file = os.path.join(test_data_dir,'rf_cb')
input_cb_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_cb\raw_CSV'
outpu_cb_train_file = os.path.join(train_data_dir,'rf_cb')

input_ws_val_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\test\rf_ws\raw_CSV'  # csv파일들이 있는 디렉토리 위치
output_ws_val_file = os.path.join(test_data_dir,'rf_ws')
input_ws_train_file = r'C:\Users\0614_\Desktop\개발용\motion_classirfy_raw_data\10. refined data\train\rf_ws\raw_CSV'
outpu_ws_train_file = os.path.join(train_data_dir,'rf_ws')

