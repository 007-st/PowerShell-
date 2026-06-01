import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# 1. 加载最原始的数据（先不加我们自己构造的交叉特征）
train_df = pd.read_csv('data_train.csv')
test_df = pd.read_csv('data_test.csv')

# 提取特征列
feature_cols = [col for col in train_df.columns if col not in ['name', 'label']]

# 2. 构造对抗验证数据集
# 训练集打标签 0，测试集打标签 1
train_adv = train_df[feature_cols].copy()
train_adv['is_test'] = 0

test_adv = test_df[feature_cols].copy()
test_adv['is_test'] = 1

# 合并数据
adv_data = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)

X_adv = adv_data[feature_cols]
y_adv = adv_data['is_test']

# 3. 训练二分类对抗模型
adv_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'seed': 2026,
    'n_jobs': -1,
    'verbose': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
oof_preds = np.zeros(len(adv_data))
feature_importances = pd.DataFrame()
feature_importances['feature'] = feature_cols

print("开始进行对抗验证 (Adversarial Validation)...")

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_adv, y_adv)):
    X_train_fold, y_train_fold = X_adv.iloc[train_idx], y_adv.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_adv.iloc[valid_idx], y_adv.iloc[valid_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

    model = lgb.train(
        adv_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    oof_preds[valid_idx] = model.predict(X_valid_fold, num_iteration=model.best_iteration)

    # 记录特征重要性
    feature_importances[f'fold_{fold + 1}'] = model.feature_importance(importance_type='gain')

# 4. 评估对抗验证结果
auc_score = roc_auc_score(y_adv, oof_preds)
print(f"\n---> 对抗验证 OOF AUC: {auc_score:.5f}")

if auc_score > 0.7:
    print("警告：AUC 远高于 0.5，存在严重的分布偏移！")
elif auc_score > 0.6:
    print("提示：AUC 偏高，存在中等程度的分布偏移。")
else:
    print("良好：AUC 接近 0.5，训练集和测试集分布基本一致。")

# 5. 打印引发分布偏移的“内鬼特征”
feature_importances['average'] = feature_importances[[f'fold_{i + 1}' for i in range(5)]].mean(axis=1)
top_features = feature_importances.sort_values(by='average', ascending=False).reset_index(drop=True)

print("\n=== 导致分布偏移的 Top 10 '内鬼特征' ===")
print("(这些特征在训练集和测试集里表现截然不同，训练分类模型时应当考虑剔除或进行强力平滑)")
print(top_features[['feature', 'average']].head(10))

# 输出：
# 开始进行对抗验证 (Adversarial Validation)...
#
# ---> 对抗验证 OOF AUC: 0.75138
# 警告：AUC 远高于 0.5，存在严重的分布偏移！
#
# === 导致分布偏移的 Top 10 '内鬼特征' ===
# (这些特征在训练集和测试集里表现截然不同，训练分类模型时应当考虑剔除或进行强力平滑)
#                         feature       average
# 0      extension_import_profile  26153.791033
# 1  identifier_variation_profile  13079.676168
# 2      parameter_block_presence  12476.597954
# 3    credential_runtime_profile  12307.803729
# 4          function_scope_level  11876.361073
# 5          pipeline_usage_level  11496.894504
# 6         task_registry_profile   8725.795462
# 7      structure_rhythm_profile   7990.924076
# 8      content_encoding_profile   6722.923040
# 9              loop_scope_level   5830.948127