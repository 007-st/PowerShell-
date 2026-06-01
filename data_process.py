import pandas as pd

# 加载训练集
train_df = pd.read_csv('data_train.csv')

# 打印 0, 1, 2 类别的精确样本数量
print("=== 训练集类别分布 ===")
print(train_df['label'].value_counts().sort_index())

