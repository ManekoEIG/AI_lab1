import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from utilities import visualize_classifier

X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5],
              [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

for C in (1, 100):
    # В sklearn >= 1.8 solver='liblinear' больше не делает one-vs-rest сам,
    # поэтому схему OvR задаём явно — модель та же, что в методичке.
    classifier = OneVsRestClassifier(
        linear_model.LogisticRegression(solver='liblinear', C=C))
    classifier.fit(X, y)
    acc = (classifier.predict(X) == y).mean() * 100
    print(f"\nC = {C}")
    print("Training accuracy = {:.2f} %".format(acc))
    print("Coefficients (one-vs-rest):\n", np.round(np.vstack([e.coef_ for e in classifier.estimators_]), 3))
    print("Intercepts:", np.round(np.hstack([e.intercept_ for e in classifier.estimators_]), 3))
    print("||w|| per class:", np.round(np.linalg.norm(np.vstack([e.coef_ for e in classifier.estimators_]), axis=1), 3))
    visualize_classifier(classifier, X, y,
                         title=f"Логистическая регрессия, C = {C}",
                         save_path=f"output/logreg_C{C}.png")
