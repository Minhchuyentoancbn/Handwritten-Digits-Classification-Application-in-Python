import pandas as pd
from sklearn.model_selection import StratifiedKFold

def run(n_fold):
    """
    Creat k-fold stratified dataset
    """
    df_train = pd.read_csv('../data/mnist/train.csv')
    X_train = df_train.drop(columns='label')
    y_train = df_train['label'].copy()

    splitter = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(splitter.split(X_train.values, y_train.values)):
        df_train.loc[test_index, 'fold'] = i + 1

    df_train.to_csv(f'../data/mnist/train_{n_fold}folds.csv', index=False)
    
    
if __name__ == '__main__':
    run(20)
    run(7)
    run(5)