import time
import warnings
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

input_file = 'income_data.txt'

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
print("Samples: <=50K =", count_class1, "| >50K =", count_class2)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


def encode_point(point):
    """Кодирование одной точки данных теми же кодировщиками."""
    encoded = [-1] * len(point)
    count = 0
    for i, item in enumerate(point):
        if item.isdigit():
            encoded[i] = int(point[i])
        else:
            encoded[i] = int(label_encoder[count].transform([point[i]])[0])
            count += 1
    return np.array(encoded).reshape(1, -1)


# ---------- Классификатор из методички ----------
print("\n=== A. OneVsOneClassifier(LinearSVC), raw features ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score (3-fold CV, weighted): " + str(round(100 * f1.mean(), 2)) + "%")
print("Hold-out accuracy:  {:.2f}%".format(100 * accuracy_score(y_test, y_test_pred)))
print("Hold-out precision: {:.2f}%".format(100 * precision_score(y_test, y_test_pred, average='weighted', zero_division=0)))
print("Hold-out recall:    {:.2f}%".format(100 * recall_score(y_test, y_test_pred, average='weighted')))
print(classification_report(y_test, y_test_pred, target_names=label_encoder[-1].classes_, zero_division=0))

# ---------- Ядра и параметр C ----------
print("\n=== B. Kernels and C (standardized features) ===")
scaler = preprocessing.StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

X_sub, _, y_sub, _ = train_test_split(Xs_train, y_train, train_size=8000,
                                      stratify=y_train, random_state=0)

configs = [
    ("LinearSVC (full train)", lambda: LinearSVC(C=1.0, random_state=0, max_iter=5000), True),
    ("SVC linear, C=1", lambda: SVC(kernel='linear', C=1.0), False),
    ("SVC rbf, C=0.1", lambda: SVC(kernel='rbf', C=0.1, gamma='scale'), False),
    ("SVC rbf, C=1", lambda: SVC(kernel='rbf', C=1.0, gamma='scale'), False),
    ("SVC rbf, C=10", lambda: SVC(kernel='rbf', C=10.0, gamma='scale'), False),
    ("SVC poly deg=3, C=1", lambda: SVC(kernel='poly', degree=3, C=1.0, gamma='scale'), False),
    ("SVC sigmoid, C=1", lambda: SVC(kernel='sigmoid', C=1.0, gamma='scale'), False),
]

rows = []
models = {}
pos = 1  # класс "">50K"
for name, make, full in configs:
    model = make()
    t0 = time.time()
    if full:
        model.fit(Xs_train, y_train)
    else:
        model.fit(X_sub, y_sub)
    pred = model.predict(Xs_test)
    dt = time.time() - t0
    rows.append((name,
                 accuracy_score(y_test, pred),
                 precision_score(y_test, pred, average='weighted', zero_division=0),
                 recall_score(y_test, pred, average='weighted'),
                 f1_score(y_test, pred, average='weighted'),
                 precision_score(y_test, pred, pos_label=pos, zero_division=0),
                 recall_score(y_test, pred, pos_label=pos),
                 dt))
    models[name] = model

print("{:<24}{:>8}{:>8}{:>8}{:>8}{:>10}{:>10}{:>8}".format(
    "model", "Acc", "Prec_w", "Rec_w", "F1_w", "Prec>50K", "Rec>50K", "time,s"))
for r in rows:
    print("{:<24}{:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>10.2f}{:>10.2f}{:>8.1f}".format(
        r[0], *(100 * v for v in r[1:7]), r[7]))

# ---------- Прогноз для отдельных точек ----------
print("\n=== C. Predictions for individual data points ===")
points = {
    "P1 (из методички)": ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
                          'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
                          '0', '0', '40', 'United-States'],
    "P2": ['52', 'Self-emp-inc', '287927', 'Masters', '14', 'Married-civ-spouse',
           'Exec-managerial', 'Husband', 'White', 'Male', '15024', '0', '60', 'United-States'],
    "P3": ['22', 'Private', '201490', 'Some-college', '10', 'Never-married',
           'Sales', 'Own-child', 'White', 'Female', '0', '0', '20', 'United-States'],
    "P4": ['44', 'Federal-gov', '178417', 'Doctorate', '16', 'Married-civ-spouse',
           'Prof-specialty', 'Wife', 'Asian-Pac-Islander', 'Female', '0', '0', '45', 'India'],
    "P5": ['39', 'Private', '160123', 'Bachelors', '13', 'Divorced',
           'Adm-clerical', 'Unmarried', 'Black', 'Female', '0', '0', '40', 'United-States'],
}
best_name = max(rows, key=lambda r: r[4])[0]
best_model = models[best_name]
print("Best kernel model by weighted F1:", best_name)
for name, p in points.items():
    enc = encode_point(p)
    pred_a = label_encoder[-1].inverse_transform(classifier.predict(enc))[0]
    pred_b = label_encoder[-1].inverse_transform(best_model.predict(scaler.transform(enc)))[0]
    print(f"{name}: {', '.join(p)}")
    print(f"    LinearSVC (методичка): {pred_a} | {best_name}: {pred_b}")
