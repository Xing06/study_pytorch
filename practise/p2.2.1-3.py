import os
import torch
import pandas as pd

os.makedirs(os.path.join('D:/study/study_pytorch/practise', 'data'), exist_ok=True)
data_file = os.path.join('D:/study/study_pytorch/practise', 'data', 'house_tiny_p.csv')
with open(data_file, 'w') as f:
    f.write('id,MSSubClass,MSZoning,LotFrontage,NumRooms,Alley,Price\n')  # 列名
    f.write('1,60,RL,65,NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,70,NA,70,2,NA,106000\n')
    f.write('3,NA,NA,80,4,NA,178100\n')
    f.write('4,NA,NA,90,NA,NA,140000\n')
    
original_data = pd.read_csv(data_file)
print(original_data)
# original_data.isna().sum()[original_data.isna().sum() == original_data.isna().sum().max()].index.tolist() 找到最大值的列名
# print(original_data.isna().sum()[3])
# print(original_data.isna().sum()[0])
# print(original_data.isna().sum())
# print(original_data.isna().sum().max())
print(original_data.isna().sum()[original_data.isna().sum() == original_data.isna().sum().max()].index.tolist())
data = original_data.drop(original_data.isna().sum()[original_data.isna().sum() == original_data.isna().sum().max()].index.tolist(), axis=1)
print(data)
# 将Alley列转换成one-hot编码
data2 = pd.get_dummies(original_data, dummy_na=True)
print(data2)
# 将数据集转换为张量格式
data2=torch.tensor(data.values)
print(data2)



