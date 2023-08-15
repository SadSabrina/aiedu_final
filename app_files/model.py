import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, balanced_accuracy_score

def get_data():

    data = pd.read_csv('df_significant.csv', index_col=0)

    column_names = list(data.columns.drop('Revenue'))

    X = data.drop('Revenue', axis=1)
    X = pd.DataFrame(np.log(X).replace([-np.inf], 0),
                     columns=column_names)

    y = data['Revenue'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train.values, X_test.values, y_train, y_test, column_names



def train_gnb(ret=False):

    X_train, X_test, y_train, y_test, column_names = get_data()

    gnb = GaussianNB(priors =[0.5, 0.5], var_smoothing=1e-13)
    gnb.fit(X_train, y_train)

    print('GNB is trained.')

    predictions = gnb.predict(X_test)

    # Оценка качества
    clf_report = classification_report(y_test, predictions)

    cv = cross_validate(gnb, X_train, y_train, cv=5, scoring='balanced_accuracy')
    balanced_accuracy_val = cv['test_score'].mean()
    balanced_accuracy_test = balanced_accuracy_score(y_test, predictions)

    print(clf_report)
    print(f'Mean val balanced accuracy (5 folds): {balanced_accuracy_val}')
    print(f'Test balanced Accuracy: {balanced_accuracy_test}')

    pkl_filename = "gaussianNB.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(gnb, file)

    print('Model saved to current directory')

    if ret:

        return gnb, X_train, y_train, X_test, y_test, column_names

    else:
        return 0


if __name__ == '__main__':

    train_gnb()


