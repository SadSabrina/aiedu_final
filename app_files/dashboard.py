from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from model import train_gnb
import pandas as pd

feature_descriptions = {'Administrative': 'The number of visited pages from the `Administrative` category.',
                'Informational' : 'The number of visited pages from the `Informational` category.',
                'ProductRelated': 'The number of visited pages from the `ProductRelated` category',
                'BounceRates' : 'The percentage of the number of sessions that launch only one request to the Google Analytics server.' ,
                'PageValues': 'Average value for a page that a user visited before landing on the goal page or completing an E commerce transaction (or both).',
                'SpecialDay' : 'A specially calculated function indicates the proximity of the site visit time to a certain holiday, on which sessions are more likely to end with a transaction.'}

if __name__ == '__main__':

    gnb, X_train, y_train, X_test, y_test, column_names = train_gnb(ret=True)
    # Отключен Warning связанный с обучением модели на данных в виде np.array и pd.DataFrame
    explainer = ClassifierExplainer(model=gnb,
                                    X=pd.DataFrame(X_test,
                                                   columns=column_names),
                                    y=y_test,
                                    X_background=pd.DataFrame(X_train,
                                                              columns=column_names)[
                                                 :200],
                                    shap='kernel',
                                    labels=['Revenue = 0', 'Revenue = 1'],
                                    descriptions=feature_descriptions
                                    )

    db = ExplainerDashboard(explainer, model_summary=True)

    db.to_yaml("dashboard.yaml", explainerfile='explainer.dill', dump_explainer=True)

    print('Dashboard is saved')

