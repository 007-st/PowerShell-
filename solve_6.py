# score: 0.71525
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与核心对齐
# ==========================================
train_df = pd.read_csv('data_train.csv')
test_df = pd.read_csv('data_test.csv')

shifted_features = [
    'extension_import_profile',
    'identifier_variation_profile',
    'parameter_block_presence'
]

print("1. 正在执行分布对齐 (Quantile Transformer)...")
for col in shifted_features:
    qt = QuantileTransformer(n_quantiles=1000, random_state=2026, output_distribution='uniform')
    full_data = pd.concat([train_df[[col]], test_df[[col]]], axis=0)
    qt.fit(full_data)
    train_df[col] = qt.transform(train_df[[col]])
    test_df[col] = qt.transform(test_df[[col]])


# ==========================================
# 2. 自动化特征交叉
# ==========================================
def generate_features(df):
    df_feat = df.copy()
    features = [col for col in df.columns if col not in ['name', 'label']]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df_feat[f'{f1}_plus_{f2}'] = df_feat[f1] + df_feat[f2]
            df_feat[f'{f1}_div_{f2}'] = df_feat[f1] / (df_feat[f2] + 1e-5)
    return df_feat


print("2. 正在构建高级交叉特征...")
train_df = generate_features(train_df)
test_df = generate_features(test_df)

feature_cols = [col for col in train_df.columns if col not in ['name', 'label']]
X = train_df[feature_cols]
y = train_df['label']
X_test = test_df[feature_cols]

# ==========================================
# 3. 准备权重与自定义评估
# ==========================================
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))


def lgb_macro_f1(preds, train_data):
    labels = train_data.get_label()
    preds = preds.reshape(3, -1).T
    preds_class = np.argmax(preds, axis=1)
    f1 = f1_score(labels, preds_class, average='macro')
    return 'macro_f1', f1, True


lgb_params = {
    'objective': 'multiclass', 'num_class': 3, 'learning_rate': 0.005,
    'max_depth': 6, 'num_leaves': 31, 'feature_fraction': 0.7,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'lambda_l1': 0.1,
    'lambda_l2': 0.1, 'seed': 2026, 'n_jobs': -1, 'verbose': -1
}

NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2026)

# ==========================================
# 4. 阶段一：提取测试集概率
# ==========================================
print("\n3. [阶段一] 训练基准模型，提取测试集概率...")
stage1_test_preds = np.zeros((len(X_test), 3))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]

    train_weights = y_train_fold.map(class_weight_dict).values
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, weight=train_weights)
    valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

    model = lgb.train(
        lgb_params, train_data, num_boost_round=2000,
        valid_sets=[train_data, valid_data], feval=lgb_macro_f1,
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    stage1_test_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS

# ==========================================
# 5. 阶段二：绝对平衡伪标签增强
# ==========================================
N_TOP = 300
pseudo_indices = []
pseudo_labels = []

print(f"\n4. [阶段二] 执行平衡伪标签提取 (单类配额 {N_TOP} 个)...")
for c in range(3):
    class_probs = stage1_test_preds[:, c]
    top_indices = np.argsort(class_probs)[::-1][:N_TOP]
    pseudo_indices.extend(top_indices)
    pseudo_labels.extend([c] * N_TOP)

pseudo_X = X_test.iloc[pseudo_indices].copy()
pseudo_y = pd.Series(pseudo_labels)
X_augmented = pd.concat([X, pseudo_X], axis=0).reset_index(drop=True)
y_augmented = pd.concat([y, pseudo_y], axis=0).reset_index(drop=True)

# ==========================================
# 6. 阶段三：重训并记录纯净 OOF
# ==========================================
print("\n5. [阶段三] 增强重训，并执行 OOF 预测...")
skf_aug = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=2026)
final_test_preds = np.zeros((len(X_test), 3))

# 注意：为了后续搜索阈值，我们只保留原始训练集部分的 OOF，不包含伪标签部分，防止过拟合
orig_oof_preds = np.zeros((len(X), 3))

for fold, (train_idx, valid_idx) in enumerate(skf_aug.split(X_augmented, y_augmented)):
    X_train_fold, y_train_fold = X_augmented.iloc[train_idx], y_augmented.iloc[train_idx]
    X_valid_fold, y_valid_fold = X_augmented.iloc[valid_idx], y_augmented.iloc[valid_idx]

    train_weights = y_train_fold.map(class_weight_dict).values
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold, weight=train_weights)
    valid_data = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=train_data)

    model = lgb.train(
        lgb_params, train_data, num_boost_round=20000,
        valid_sets=[train_data, valid_data], feval=lgb_macro_f1,
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )

    # 提取当前验证集中属于【原始训练集】的索引，并记录预测概率
    orig_valid_mask = valid_idx < len(X)
    orig_valid_idx_real = valid_idx[orig_valid_mask]
    if len(orig_valid_idx_real) > 0:
        orig_oof_preds[orig_valid_idx_real] = model.predict(X_augmented.iloc[valid_idx][orig_valid_mask],
                                                            num_iteration=model.best_iteration)

    final_test_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS

# ==========================================
# 7. 宏平均阈值校准
# ==========================================
print("\n6. [阶段四] 启动 Nelder-Mead 算法，搜索极致概率乘子...")


def f1_opt_func(weights, oof_probs, y_true):
    adjusted_probs = oof_probs * weights
    preds = np.argmax(adjusted_probs, axis=1)
    return -f1_score(y_true, preds, average='macro')


base_f1 = f1_score(y, np.argmax(orig_oof_preds, axis=1), average='macro')
print(f"   校准前原始 OOF Macro F1: {base_f1:.5f}")

res = minimize(
    f1_opt_func, [1.0, 1.0, 1.0], args=(orig_oof_preds, y),
    method='Nelder-Mead', options={'maxiter': 500}
)

best_weights = res.x
opt_f1 = f1_score(y, np.argmax(orig_oof_preds * best_weights, axis=1), average='macro')
print(f"   找到的完美概率权重: {best_weights}")
print(f"   校准后极致 OOF Macro F1: {opt_f1:.5f}")

# ==========================================
# 8. 生成终极提交文件
# ==========================================
# 将搜索到的完美权重应用到测试集
final_calibrated_preds = np.argmax(final_test_preds * best_weights, axis=1)

submission = pd.DataFrame({
    'name': test_df['name'] if 'name' in test_df.columns else range(len(test_df)),
    'label': final_calibrated_preds
})
submission.to_csv('submission.csv', index=False, encoding='utf-8')
print("\n 已生成提交文件: submission.csv")

# 1. 正在执行分布对齐 (Quantile Transformer)...
# 2. 正在构建高级交叉特征...
#
# 3. [阶段一] 训练基准模型，提取测试集概率...
#
# 4. [阶段二] 执行平衡伪标签提取 (单类配额 300 个)...
#
# 5. [阶段三] 增强重训，并执行 OOF 预测...
#
# 6. [阶段四] 启动 Nelder-Mead 算法，搜索极致概率乘子...
#    校准前原始 OOF Macro F1: 0.70940
#    找到的完美概率权重: [1.013057   0.98824103 0.99783473]
#    校准后极致 OOF Macro F1: 0.74425
#
#  已生成终局校准提交文件: submission.csv

