import os
import torch
import pandas as pd
print(os.getcwd())  
os.makedirs(os.path.join('D:/study/study_pytorch/practise', 'data'), exist_ok=True)
print(os.path.join('D:/study/study_pytorch/practise', 'data'))
data_file = os.path.join('D:/study/study_pytorch/practise', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
print(outputs)
inputs = inputs.fillna(inputs.mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# x = torch.tensor(inputs.to_numpy(dtype=float))
# y = torch.tensor(outputs.to_numpy(dtype=float))
# print(x)
# print(y)
x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x)
print(y)