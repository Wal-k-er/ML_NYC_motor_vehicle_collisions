import warnings

import numpy as np
import pandas as pd
import seaborn as sb
import category_encoders as ce
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot as plt
from sklearn import exceptions

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import compute_class_weight
from sklearn.linear_model import Perceptron

warnings.filterwarnings("ignore", category=exceptions.UndefinedMetricWarning)

train = pd.read_csv("NYC_Motor_Vehicle_Collisions_to_Person.csv", encoding="utf-8", engine='python')
train_copy = train.copy()

train_copy = train_copy.drop(['VEHICLE_ID', 'PERSON_ID', 'UNIQUE_ID', 'COLLISION_ID'], axis=1)

train_copy.columns = (train_copy.columns
                      .str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True)
                      .str.lower())
#
#   Количество признаков
#
# отбор числовых колонок
columns_numeric = train_copy.select_dtypes(include=[np.number])
numeric_names = columns_numeric.columns.values
print(f'Числовые значения:{numeric_names}')

# отбор нечисловых колонок
columns_non_numeric = train_copy.select_dtypes(exclude=[np.number])
non_numeric_names = columns_non_numeric.columns.values
print(f'Нечисловые значения:{non_numeric_names}')

print(train_copy.isnull().sum())
# Графики для поиска выбросов
# for name in train_copy.columns:
#     print(train_copy[name].value_counts())
#     sb.countplot(train_copy[name])
#
#     plt.show()
#
# for name in train_copy.columns:
#     sb.boxplot(train_copy[name])
#     plt.show()

# for name in train_copy.columns:
#     sb.scatterplot(data=train_copy,x=train_copy['person_injury'], y=train_copy[name])
#     plt.show()

train_copy = train_copy.loc[train_copy['person_sex'] != 'U']

train_copy = train_copy.fillna({'safety_equipment': 'Unknown',
                                'ped_location': 'Unknown',
                                'ejection': 'Unknown',
                                'position_in_vehicle': 'Unknown'})

train_copy = train_copy.dropna(subset=['person_age', 'contributing_factor_1', 'contributing_factor_2'])

train_copy.loc[train_copy['contributing_factor_1']
                   .isin(train_copy['contributing_factor_1']
                         .value_counts()
                         .index[train_copy['contributing_factor_1']
                         .value_counts() < 10]), 'contributing_factor_1'] = 'Other'

train_copy.loc[train_copy['contributing_factor_2'] \
                   .isin(train_copy['contributing_factor_2']
                         .value_counts()
                         .index[train_copy['contributing_factor_2']
                         .value_counts() < 10]), 'contributing_factor_2'] = 'Other'

print(train_copy['contributing_factor_1'].value_counts())
print(train_copy['contributing_factor_2'].value_counts())

train_copy.loc[train_copy['person_age'] < 0.0] = train_copy['person_age'].mean()
train_copy.loc[train_copy['person_age'] > 100.0] = train_copy['person_age'].mean()

train_copy['age_category'] = pd.cut(train_copy['person_age'], bins=[0, 26, 35, 60, np.inf],
                                    labels=['Юный (<26)', 'Взрослый (26-35)', 'Взрослый (35-60)', 'Пожилой']) \
    .astype(str)

train_copy = train_copy.loc[train_copy['age_category'] != 'nan']
print(train_copy['age_category'].value_counts())

train_copy.loc[train_copy['ped_action'] == 37.49576186787651, 'ped_action'] = 'Unknown'
train_copy.loc[train_copy['ejection'] == 37.49576186787651, 'ejection'] = 'Unknown'
train_copy = train_copy.loc[(train_copy['person_injury'] != 37.49576186787651) &
                            (train_copy['person_injury'] != 37.81004307999686)]

train_copy = train_copy.loc[(train_copy['ped_role'] != 41.93902849121094) &
                            (train_copy['ped_role'] != 41.78234375)]
train_copy = train_copy.loc[(train_copy['crash_time'] != 41.93902849121094) &
                            (train_copy['crash_time'] != 41.78234375)]
train_copy = train_copy.loc[(train_copy['crash_date'] != 41.93902849121094) &
                            (train_copy['crash_date'] != 41.78234375)]
train_copy = train_copy.loc[(train_copy['position_in_vehicle'] != 41.93902849121094) &
                            (train_copy['position_in_vehicle'] != 41.78234375) &
                            (train_copy['position_in_vehicle'] != 37.49576186787651)]

train_copy['crash_date'] = pd.to_datetime(train_copy['crash_date'], format="%Y-%m-%d")
train_copy['crash_time'] = pd.to_datetime(train_copy['crash_time'], format="%H:%M")

# графики

sb.countplot(x='person_injury', data=train_copy)
plt.show()

fig, ((fig1, fig2), (fig3, fig4)) = plt.subplots(2, 2, figsize=(5, 15))

fig1 = sb.countplot(x='person_sex', hue='person_injury', data=train_copy, ax=fig1)

fig2 = sb.countplot(x='bodily_injury', hue='person_injury', data=train_copy, ax=fig2)
fig2.set_xticklabels(fig2.get_xticklabels(), rotation=20, ha='right')
fig3 = sb.countplot(x='person_type', hue='person_injury', data=train_copy, ax=fig3)

fig4 = sb.countplot(x='age_category', hue='person_injury', data=train_copy, ax=fig4)

plt.tight_layout()

plt.show()

train_copy['person_sex'] = train_copy['person_sex'].replace({'F': 0, 'M': 1})


def get_int_date(date_str):
    return int(date_str.timestamp())


train_copy['crash_date'] = train_copy['crash_date'].apply(get_int_date)


def get_hours(time_str):
    hour = time_str.hour
    return int(hour)


train_copy['crash_time'] = train_copy['crash_time'].apply(get_hours)

# for name in train_copy.columns:
#     sb.catplot(data=train_copy,x='person_injury',y=name)
#     plt.show()

train_copy['person_injury'].replace({'Killed': 0, 'Injured': 1}, inplace=True)

train_copy = train_copy.drop('age_category', axis=1)
train_ohe = pd.get_dummies(train_copy, columns=['ped_role', 'ped_location', 'ped_action',
                                                'safety_equipment', 'person_type', 'contributing_factor_1',
                                                'contributing_factor_2', 'bodily_injury', 'ejection', 'complaint',
                                                'emotional_status', 'position_in_vehicle'])

cats = []

for col in train_copy.columns:
    if train_copy[col].dtype == 'object':
        cats.append(col)

ord_enc = ce.OrdinalEncoder(cols=cats, handle_unknown='impute')
ord_enc_train = ord_enc.fit_transform(train_copy)

# for name in train_copy.columns:
#     sb.boxplot(ord_enc_train[name])
#     plt.show()

#   Heatmap
# corrmat = ord_enc_train.corr()
#
# f, ax = plt.subplots(figsize=(14,14))
#
# sb.heatmap(corrmat,vmax=.8,square=True)
# plt.show()
#
# cols = corrmat.nlargest(10,'person_injury')['person_injury'].index
# cm=np.corrcoef(ord_enc_train[cols].values.T)
# hm = sb.heatmap(cm, cbar=True, annot=True, square=True,fmt='.2f',yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
#
# most_corr = pd.DataFrame(cols)
# most_corr.columns=['Самые связанные']
# print(most_corr)

# cols = corrmat.nlargest(15, 'person_injury')['person_injury'].index
# cm = np.corrcoef(train_ohe[cols].values)
# hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
#
# most_corr = pd.DataFrame(cols)
# most_corr.columns = ['Самые связанные']
# print(most_corr)

X = train_ohe.drop('person_injury', axis=1)
Y = train_ohe['person_injury']

X_ord = ord_enc_train.drop('person_injury', axis=1)
Y_ord = ord_enc_train['person_injury']


# Метод взвешивания классов
def get_class_weights(y_not_balanced):
    return compute_class_weight('balanced', classes=np.unique(y_not_balanced), y=y_not_balanced)


# Метод недостаточной выборки (случайное удаление и удаление схожих со строками класса-меньшинства )
def balance_train(x_for_balance, y_for_balance):
    nm = NearMiss()
    return nm.fit_resample(x_for_balance, y_for_balance.ravel())


# Метод увеличения выборки посредством копирования из класса-меньшинства
def get_over_sampled(over_x, over_y):
    ros = RandomOverSampler()
    return ros.fit_resample(over_x, over_y)


def get_train_test(x, y):
    return train_test_split(x, y, test_size=0.33, random_state=42)


def get_metrics(test_x, test_y, pred_y, best_model):
    confmat = confusion_matrix(y_true=test_y, y_pred=pred_y)
    specificity = confmat[1][1] / (confmat[1][0] + confmat[1][1])
    sensitivity = confmat[0][0] / (confmat[0][0] + confmat[0][1])
    print(r'Чувствительность: %.3f' % sensitivity)
    print(r'Специфичность: %.3f' % specificity)
    preds = best_model.predict(test_x)

    fpr, tpr, threshold = roc_curve(test_y, preds)
    roc_auc = auc(fpr, tpr)
    print(r'AUC: %.3f' % roc_auc)
    return sensitivity, specificity, roc_auc


def logistic_regression(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print('Логистическая регрессия')
    if weigts is -1:
        model = LogisticRegression()
    else:
        model = LogisticRegression(max_iter=4000, class_weight=dict(enumerate(weigts)))

    parameters = {
        'solver': ['lbfgs', 'liblinear'],
        'C': [0.5, 1.0, 5.0, 9.0, 10.0],
        'fit_intercept': [True, False]
    }

    grid_logistic = GridSearchCV(estimator=model, param_grid=parameters)
    grid_logistic.fit(df_x, df_y)
    accuracy = round(grid_logistic.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid_logistic.best_params_}')
    best_logistic_model = grid_logistic.best_estimator_
    y_pred_grid = best_logistic_model.predict(df_test_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_grid))

    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred_grid, best_logistic_model)

    return accuracy, sens, spec, roc


def tree_classifier(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print('Дерево решений')

    if weigts is -1:
        model = DecisionTreeClassifier()

    else:
        model = DecisionTreeClassifier(class_weight=dict(enumerate(weigts)))

    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [10, 12, 14, 16],
                  'min_samples_split': [10, 12, 14, 16]}

    grid_tree = GridSearchCV(estimator=model, param_grid=parameters)
    grid_tree.fit(df_x, df_y)
    accuracy = round(grid_tree.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid_tree.best_params_}')
    best_tree_model = grid_tree.best_estimator_
    y_pred_tree_grid = best_tree_model.predict(df_test_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_tree_grid))

    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred_tree_grid, best_tree_model)

    return accuracy, sens, spec, roc


def svc(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print('Метод опорных векторов')
    if weigts is -1:
        model = SVC()
    else:
        model = SVC(class_weight=dict(enumerate(weigts)))

    model.fit(df_x, df_y)

    y_pred = model.predict(df_test_x)

    accuracy = round(accuracy_score(df_test_y, y_pred), 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred, model)

    return accuracy, sens, spec, roc


def gradient_boosting(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print("\nГрадиентный бустинг")

    model = GradientBoostingClassifier()

    parameters = {
        'learning_rate': [0.2, 0.7],
        'max_depth': [10, 15],
        'min_samples_split': [10, 15]
    }

    grid = GridSearchCV(estimator=model, param_grid=parameters)
    if weigts is -1:
        grid.fit(df_x, df_y)
    else:
        weigts = np.array(weigts)
        sample_weights = np.zeros(len(df_y))
        sample_weights[df_y == 0] = weigts[0]
        sample_weights[df_y == 1] = weigts[1]
        grid.fit(df_x, df_y, sample_weight=sample_weights)

    accuracy = round(grid.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid.best_params_}')
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(df_test_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_grid))

    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred_grid, best_model)

    return accuracy, sens, spec, roc


def perceptron(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print('Перцептрон')

    if weigts is -1:
        model = Perceptron()
    else:
        model = Perceptron(class_weight=dict(enumerate(weigts)))

    parameters = {
        'eta0': [0.1, 0.2, 0.3],
        'penalty': ['l2', 'l1']
    }

    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(df_x, df_y)
    accuracy = round(grid.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid.best_params_}')
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(df_test_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_grid))

    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred_grid, best_model)

    return accuracy, sens, spec, roc


def random_forest(df_x, df_text_x, df_y, df_test_y, weigts=-1):
    print('Случайный лес')

    if weigts is -1:
        model = RandomForestClassifier()
    else:
        model = RandomForestClassifier(class_weight=dict(enumerate(weigts)))

    parameters = {
        'n_estimators': [70, 80, 90, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 12, 14, 16],
        'min_samples_split': [10, 12, 14, 16]
    }

    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(df_x, df_y)
    accuracy = round(grid.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid.best_params_}')
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(df_text_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_grid))

    sens, spec, roc = get_metrics(df_text_x, df_test_y, y_pred_grid, best_model)

    return accuracy, sens, spec, roc


def hist_gradient(df_x, df_test_x, df_y, df_test_y, weigts=-1):
    print("\nHist Gradient")

    if weigts is -1:
        model = HistGradientBoostingClassifier()
    else:
        model = HistGradientBoostingClassifier(class_weight=dict(enumerate(weigts)))

    parameters = {
        'learning_rate': [0.4, 0.7, 1.0],
        'max_depth': [10, 12, 14]
    }

    grid = GridSearchCV(estimator=model, param_grid=parameters)
    grid.fit(df_x, df_y)
    accuracy = round(grid.best_score_, 3)
    print(f'Верность (Acc) лучшей модели: {accuracy}')
    print(f'Лучшие параметры модели: {grid.best_params_}')
    best_model = grid.best_estimator_
    y_pred_grid = best_model.predict(df_test_x)
    print(classification_report(y_true=df_test_y, y_pred=y_pred_grid))
    sens, spec, roc = get_metrics(df_test_x, df_test_y, y_pred_grid, best_model)

    return accuracy, sens, spec, roc


def plot_metrics(accuracies, sensivities, specifities, aucs, label, plot_num):
    x = np.arange(len(accuracy_array))
    width = 0.2
    plt.subplot(3, 1, plot_num)
    plt.bar(x - 0.4, accuracies, width=width)
    plt.bar(x - 0.2, sensivities, width=width)
    plt.bar(x, specifities, width=width)
    plt.bar(x + 0.2, aucs, width=width)

    plt.xlabel(label)
    plt.xticks(x, ['Логистическая регрессия', 'Дерево решений', 'Перцептрон', 'SVM', 'Hist градиент',
                   'Градиентный бустинг', 'Случайный лес'])
    plt.legend(['accuracy',
                'sensivity',
                'specifity',
                'auc'])


def teach_models(train_x, test_x, train_y, test_y, weights=-1):
    acc_array = [0] * 7
    sens_array = [0] * 7
    spec_array = [0] * 7
    roc_auc_array = [0] * 7
    acc_array[0], sens_array[0], spec_array[0], roc_auc_array[0] = \
        logistic_regression(train_x, test_x,
                            train_y, test_y, weights)

    acc_array[1], sens_array[1], spec_array[1], roc_auc_array[1] = \
        tree_classifier(train_x, test_x,
                        train_y, test_y, weights)

    acc_array[2], sens_array[2], spec_array[2], roc_auc_array[2] = \
        perceptron(train_x, test_x,
                   train_y, test_y, weights)

    acc_array[3], sens_array[3], spec_array[3], roc_auc_array[3] = \
        svc(train_x, test_x,
            train_y, test_y, weights)

    acc_array[4], sens_array[4], spec_array[4], roc_auc_array[4] = \
        hist_gradient(train_x, test_x,
                      train_y, test_y, weights)

    acc_array[5], sens_array[5], spec_array[5], roc_auc_array[5] = \
        gradient_boosting(train_x, test_x,
                          train_y, test_y, weights)

    acc_array[6], sens_array[6], spec_array[6], roc_auc_array[6] = \
        random_forest(train_x, test_x,
                      train_y, test_y, weights)
    return acc_array, sens_array, spec_array, roc_auc_array


label = 'One hot encoding балансированные NearMiss'
print(label)

train_x, test_x, train_y, test_y = get_train_test(X, Y)

train_balanced_x, train_balanced_y = balance_train(train_x, train_y)

accuracy_array, sensivity_array, specifity_array, auc_array = \
    teach_models(train_balanced_x, test_x,
                 train_balanced_y, test_y)

plot_metrics(accuracy_array, sensivity_array, specifity_array, auc_array, label, 1)

label = 'One hot encoding балансированные OverSampling'
print(label)

train_over_x, test_over_x, train_over_y, test_over_y = get_train_test(X, Y)

train_over_x, train_over_y = get_over_sampled(train_over_x, train_over_y)

accuracy_array, sensivity_array, specifity_array, auc_array = \
    teach_models(train_over_x, test_over_x,
                 train_over_y, test_over_y)

plot_metrics(accuracy_array, sensivity_array, specifity_array, auc_array, label, 2)

label = 'Ordinal Encoder с использованием взвешивания классов'
print(label)

train_ord_x, test_ord_x, train_ord_y, test_ord_y = get_train_test(X_ord, Y_ord)

weights = get_class_weights(train_ord_y)

accuracy_array, sensivity_array, specifity_array, auc_array = \
    teach_models(train_over_x, test_over_x,
                 train_over_y, test_over_y, weights)

plot_metrics(accuracy_array, sensivity_array, specifity_array, auc_array, label, 3)

plt.show()
