import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def get_base_info(x):
    return [i.split(':')[-1] for i in x.split(' ')]


def get_speed(x):
    return np.array([i.split(',')[0] for i in x], dtype='float16')


def get_eta(x):
    return np.array([i.split(',')[1] for i in x], dtype='float16')


def get_state(x):
    return [int(i.split(',')[2]) for i in x]


def get_cnt(x):
    return np.array([i.split(',')[3] for i in x], dtype='int16')


def gen_feats(path, mode='is_train'):
    df = pd.read_csv(path, sep=';', header=None)
    df['link'] = df[0].apply(lambda x: x.split(' ')[0])
    if mode == 'is_train':
        df['label'] = df[0].apply(lambda x: int(x.split(' ')[1]))
        df['label'] = df['label'].apply(lambda x: 3 if x > 3 else x)
        df['label'] -= 1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))
    else:
        df['label'] = -1
        df['current_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[2]))
        df['future_slice_id'] = df[0].apply(lambda x: int(x.split(' ')[3]))

    df['time_diff'] = df['future_slice_id'] - df['current_slice_id']

    df['curr_state'] = df[1].apply(lambda x: x.split(' ')[-1].split(':')[-1])
    df['curr_speed'] = df['curr_state'].apply(lambda x: x.split(',')[0])
    df['curr_eta'] = df['curr_state'].apply(lambda x: x.split(',')[1])
    df['curr_cnt'] = df['curr_state'].apply(lambda x: x.split(',')[3])
    df['curr_state'] = df['curr_state'].apply(lambda x: x.split(',')[2])
    del df[0]

    for i in tqdm(range(1, 6)):
        df['his_info'] = df[i].apply(get_base_info)
        if i == 1:
            flg = 'current'
        else:
            flg = f'his_{(6 - i) * 7}'
        df['his_speed'] = df['his_info'].apply(get_speed)
        df[f'{flg}_speed_min'] = df['his_speed'].apply(lambda x: x.min())
        df[f'{flg}_speed_max'] = df['his_speed'].apply(lambda x: x.max())
        df[f'{flg}_speed_mean'] = df['his_speed'].apply(lambda x: x.mean())
        df[f'{flg}_speed_std'] = df['his_speed'].apply(lambda x: x.std())

        df['his_eta'] = df['his_info'].apply(get_eta)
        df[f'{flg}_eta_min'] = df['his_eta'].apply(lambda x: x.min())
        df[f'{flg}_eta_max'] = df['his_eta'].apply(lambda x: x.max())
        df[f'{flg}_eta_mean'] = df['his_eta'].apply(lambda x: x.mean())
        df[f'{flg}_eta_std'] = df['his_eta'].apply(lambda x: x.std())

        df['his_cnt'] = df['his_info'].apply(get_cnt)
        df[f'{flg}_cnt_min'] = df['his_cnt'].apply(lambda x: x.min())
        df[f'{flg}_cnt_max'] = df['his_cnt'].apply(lambda x: x.max())
        df[f'{flg}_cnt_mean'] = df['his_cnt'].apply(lambda x: x.mean())
        df[f'{flg}_cnt_std'] = df['his_cnt'].apply(lambda x: x.std())

        df['his_state'] = df['his_info'].apply(get_state)
        df[f'{flg}_state'] = df['his_state'].apply(lambda x: Counter(x).most_common()[0][0])
        df.drop([i, 'his_info', 'his_speed', 'his_eta', 'his_cnt', 'his_state'], axis=1, inplace=True)
    if mode == 'is_train':
        df.to_csv(f"{mode}_{path.split('/')[-1]}", index=False)
    else:
        df.to_csv(f"is_test.csv", index=False)


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average=None)
    scores = scores[0]*0.2+scores[1]*0.2+scores[2]*0.6
    return 'f1_score', scores, True


def lgb_train(train_: pd.DataFrame, test_: pd.DataFrame, use_train_feats: list, id_col: str, label: str,
              n_splits: int, split_rs: int, is_shuffle=True, use_cart=False, cate_cols=None) -> pd.DataFrame:
    if not cate_cols:
        cate_cols = []
    print('data shape:\ntrain--{}\ntest--{}'.format(train_.shape, test_.shape))
    print('Use {} features ...'.format(len(use_train_feats)))
    print('Use lightgbm to train ...')
    n_class = train_[label].nunique()
    train_[f'{label}_pred'] = 0
    test_pred = np.zeros((test_.shape[0], n_class))
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = use_train_feats

    folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=split_rs)
    train_user_id = train_[id_col].unique()

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_user_id), start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_.loc[train_[id_col].isin(train_user_id[train_idx]), use_train_feats], train_.loc[
            train_[id_col].isin(train_user_id[train_idx]), label]
        valid_x, valid_y = train_.loc[train_[id_col].isin(train_user_id[valid_idx]), use_train_feats], train_.loc[
            train_[id_col].isin(train_user_id[valid_idx]), label]
        print(f'for train user:{len(train_idx)}\nfor valid user:{len(valid_idx)}')

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_score_eval
        )
        fold_importance_df[f'fold_{n_fold}_imp'] = clf.feature_importance(importance_type='gain')
        train_.loc[train_[id_col].isin(train_user_id[valid_idx]), f'{label}_pred'] = np.argmax(
            clf.predict(valid_x, num_iteration=clf.best_iteration), axis=1)
        test_pred += clf.predict(test_[use_train_feats], num_iteration=clf.best_iteration) / folds.n_splits

    report = f1_score(train_[label], train_[f'{label}_pred'], average=None)
    print(classification_report(train_[label], train_[f'{label}_pred'], digits=4))
    print('Score: ', report[0] * 0.2 + report[1] * 0.2 + report[2] * 0.6)
    test_[f'{label}_pred'] = np.argmax(test_pred, axis=1)
    test_[label] = np.argmax(test_pred, axis=1)+1
    five_folds = [f'fold_{f}_imp' for f in range(1, n_splits + 1)]
    fold_importance_df['avg_imp'] = fold_importance_df[five_folds].mean(axis=1)
    fold_importance_df.sort_values(by='avg_imp', ascending=False, inplace=True)
    print(fold_importance_df[['Feature', 'avg_imp']].head(20))
    return test_[[id_col, 'current_slice_id', 'future_slice_id', label]]


if __name__ == "__main__":
    train_path = './traffic/20190730.txt'
    test_path = 'test.txt'
    gen_feats(train_path, mode='is_train')
    gen_feats(test_path, mode='is_test')
    attr = pd.read_csv('attr.txt', sep='\t',
                       names=['link', 'length', 'direction', 'path_class', 'speed_class', 'LaneNum', 'speed_limit',
                              'level', 'width'], header=None)

    train = pd.read_csv('is_train_20190730.txt')
    test = pd.read_csv('is_test.csv')
    train = train.merge(attr, on='link', how='left')
    test = test.merge(attr, on='link', how='left')

    use_cols = [i for i in train.columns if i not in ['link', 'label', 'current_slice_id', 'future_slice_id', 'label_pred']]

    sub = lgb_train(train, test, use_cols, 'link', 'label', 5, 2020)

    sub.to_csv('public_baseline.csv', index=False, encoding='utf8')