import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from time import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import RocCurveDisplay

from wrangle_data import output_cleaned_data


X_cols = [
    'age',
    'job_admin.',
    'job_blue-collar',
    'job_entrepreneur',
    'job_housemaid',
    'job_management',
    'job_retired',
    'job_self-employed',
    'job_services',
    'job_student',
    'job_technician',
    'job_unemployed',
    'job_unknown',
    'marital_divorced',
    'marital_married',
    'marital_single',
    'marital_unknown',
    'education_basic.4y',
    'education_basic.6y',
    'education_basic.9y',
    'education_high.school',
    'education_illiterate',
    'education_professional.course',
    'education_university.degree',
    'education_unknown',
    'default_no',
    'default_yes',
    'default_unknown',
    'housing_no',
    'housing_yes',
    'housing_unknown',
    'loan_no',
    'loan_yes',
    'loan_unknown',
    'contact_cellular',
    'month_mar',
    'month_apr',
    'month_may',
    'month_jun',
    'month_jul',
    'month_aug',
    'month_sep',
    'month_oct',
    'month_nov',
    'month_dec',
    'day_of_week_mon',
    'day_of_week_tue',
    'day_of_week_wed',
    'day_of_week_thu',
    'day_of_week_fri',
    'campaign',
    'previous',
    'poutcome_failure',
    'poutcome_nonexistent',
    'poutcome_success',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed',
    'total_clients_contacted_in_month'
]

y_col = 'y'


def train_random_forest(X_train, y_train, X_test, y_test, seed=0x1337beef):
    '''Train random forest classifier, saving model and creating visualization
    of permutation importance'''

    # using cross validation like the commented out code below, I determined
    # that the parameters on the uncommented line below were good choices

    # clf = RandomForestClassifier(n_estimators=100, max_features="sqrt")
    # cross_val_score(clf, X_train, y_train, cv=5)

    clf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                min_samples_split=50, random_state=seed)

    print("Training random forest classifier...\n")

    clf.fit(X_train, y_train)

    print("Random forest accuracy scores")
    print("Train: %f" % clf.score(X_train, y_train)) # 0.907618
    print("Test: %f\n" % clf.score(X_test, y_test)) # 0.904710

    # now let's compute feature importance, adapted from
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    print("Computing feature importance...\n")
    result = permutation_importance(clf, X_test, y_test, random_state=seed)

    importances = pd.DataFrame(result.importances_mean, index=X_test.columns,
                                columns=['average_decrease_in_accuracy_score'])

    # create bar chart
    importances.sort_values('average_decrease_in_accuracy_score', inplace=True)
    subset = importances[importances.average_decrease_in_accuracy_score > 0]

    fig, ax = plt.subplots(figsize=[6.0, 6.0], dpi=200)

    ax.barh(subset.index, subset['average_decrease_in_accuracy_score'])
    ax.set_ylabel('Feature')
    ax.set_xlabel('Average decrease in accuracy score')
    ax.set_title('Random Forest Permutation Importances')

    fig.tight_layout()

    if not os.path.isdir('figures'):
        os.mkdir('figures')

    fig.savefig('figures/random_forest_feature_importance.png')

    # save random forest classifier for posterity
    if not os.path.isdir('models'):
        os.mkdir('models')

    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(clf, f)

    return clf


def random_forest_predict(X, clf=None):
    '''Returns predictions based on the random forest model'''

    #if no classifier is passed, grab saved model if exists, otherwise train
    if not clf:
        if os.path.isfile('models/random_forest.pkl'):
            with open('models/random_forest.pkl', 'rb') as f:
                clf = pickle.load(f)
        else:
            clf = train_random_forest()

    return clf.predict(X)


def train_neural_net(X_train, y_train, X_test, y_test, seed=0x1337beef):
    '''Train a simple neural network classifier, saving model and creating
    visualization of permutation importance'''

    # this classifier is sensitive to feature scaling so we are going to use
    # sklearn's StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # We'll set up a simple neural net with a single hidden layer
    clf = MLPClassifier(hidden_layer_sizes=(20,), random_state=seed)

    print("Training neural network classifier...\n")

    clf.fit(X_train_scaled, y_train)

    print("Neural network accuracy scores")
    print("Train: %f" % clf.score(X_train_scaled, y_train)) # 0.907011
    print("Test: %f\n" % clf.score(X_test_scaled, y_test)) # 0.902889

    # now let's compute feature importance, adapted from
    # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
    print("Computing feature importance...\n")
    result = permutation_importance(clf, X_test_scaled, y_test, random_state=seed)

    importances = pd.DataFrame(result.importances_mean, index=X_test.columns,
                                columns=['average_decrease_in_accuracy_score'])

    # create bar chart
    importances.sort_values('average_decrease_in_accuracy_score', inplace=True)
    subset = importances[importances.average_decrease_in_accuracy_score > 0]

    fig, ax = plt.subplots(figsize=[6.0, 6.0], dpi=200)

    ax.barh(subset.index, subset['average_decrease_in_accuracy_score'])
    ax.set_ylabel('Feature')
    ax.set_xlabel('Average decrease in accuracy score')
    ax.set_title('Neural Network Permutation Importances')

    fig.tight_layout()

    if not os.path.isdir('figures'):
        os.mkdir('figures')

    fig.savefig('figures/neural_network_feature_importance.png')

    # save neural network classifier for posterity
    if not os.path.isdir('models'):
        os.mkdir('models')

    with open('models/neural_network.pkl', 'wb') as f:
        pickle.dump(clf, f)

    #we also need to save out the scaler for future use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return clf, scaler


def neural_net_predict(X, clf=None, scaler=None):
    '''Returns predictions based on the neural network model'''

    #if no classifier is passed, grab saved model if exists, otherwise train
    if not clf or not scaler:

        if os.path.isfile('models/neural_network.pkl') \
            and os.path.isfile('models/scaler.pkl'):

            with open('models/neural_network.pkl', 'rb') as f:
                clf = pickle.load(f)

            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

        else:
            clf, scaler = train_neural_net()

    return clf.predict(scaler.transform(X))



if __name__ == "__main__":

    # make sure cleaned data exists
    if not os.path.isfile('data/cleaned_dataset.csv'):
        output_cleaned_data()

    df = pd.read_csv('data/cleaned_dataset.csv', index_col=0)

    X = df[X_cols]
    y = df[y_col]

    # split data into training and testing blocks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0x1337beef)

    # save splits for posterity
    X_train.to_csv('data/X_train.csv')
    X_test.to_csv('data/X_test.csv')
    y_train.to_csv('data/y_train.csv')
    y_test.to_csv('data/y_test.csv')


    # train random forest classifier
    rf = train_random_forest(X_train, y_train, X_test, y_test)

    # train neural network classifier
    nn, scaler = train_neural_net(X_train, y_train, X_test, y_test)

    # which model executes faster?
    start = time()
    preds = random_forest_predict(X_test, rf)
    rf_done = time()
    preds = neural_net_predict(X_test, nn, scaler)
    nn_done = time()

    print("Random forest predictions ran in: %f" % (rf_done - start))
    print("Random forest predictions ran in: %f" % (nn_done - rf_done))

    # generate a ROC curve plot of both classifiers
    fig, ax = plt.subplots(figsize=[6.0, 6.0], dpi=200)

    rf_roc = RocCurveDisplay.from_estimator(rf, X_test, y_test)
    rf_roc.plot(ax=ax)
    nn_roc = RocCurveDisplay.from_estimator(nn, scaler.transform(X_test), y_test)
    nn_roc.plot(ax=ax)

    ax.set_title('Random Forest vs. Neural Net ROC Curves')
    fig.savefig('figures/roc_curves.png')
