import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

raw = np.genfromtxt('boston_housing.csv', delimiter=',', skip_header=1,
                    converters={3: lambda s: float(str(s).strip('"\' b'))})
raw = np.array([list(r) for r in raw], dtype=float)
data_X, data_y = raw[:, :13], raw[:, 13]
print("Dataset:", data_X.shape)

X, y = shuffle(data_X, data_y, random_state=7)

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)

y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print("\nPredicted price:", sv_regressor.predict([test_data])[0])

# Эксперименты с C и epsilon 
points = {
    "T1 (методичка)": test_data,
    "T2 (благополучный район)": [0.03, 20, 3.3, 0, 0.44, 7.2, 30, 6.0, 4, 280, 15.5, 395.0, 4.0],
    "T3 (средний)": [0.25, 0, 8.1, 0, 0.53, 6.1, 70, 3.8, 5, 300, 19.0, 390.0, 11.0],
    "T4 (неблагополучный)": [15.0, 0, 18.1, 0, 0.70, 5.4, 98, 1.6, 24, 666, 20.2, 50.0, 28.0],
}

print("\n#### Grid over C and epsilon (kernel='linear', raw features) ####")
hdr = "{:>6}{:>8}{:>9}{:>7}{:>6}".format("C", "eps", "MSE", "EVS", "nSV")
hdr += "".join("{:>9}".format(k.split()[0]) for k in points)
print(hdr)
results = []
for C in (0.1, 1.0, 10.0):
    for eps in (0.1, 1.0, 5.0):
        m = SVR(kernel='linear', C=C, epsilon=eps).fit(X_train, y_train)
        p = m.predict(X_test)
        preds = [m.predict([v])[0] for v in points.values()]
        results.append((C, eps, mean_squared_error(y_test, p),
                        explained_variance_score(y_test, p), len(m.support_), preds))
        print("{:>6}{:>8}{:>9.2f}{:>7.2f}{:>6}".format(C, eps, results[-1][2], results[-1][3], results[-1][4])
              + "".join("{:>9.2f}".format(v) for v in preds))

print("\n#### RBF kernel with feature scaling ####")
print(hdr)
for C in (1.0, 10.0, 100.0):
    for eps in (0.1, 1.0):
        m = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=C, epsilon=eps, gamma='scale'))
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds = [m.predict([v])[0] for v in points.values()]
        print("{:>6}{:>8}{:>9.2f}{:>7.2f}{:>6}".format(
            C, eps, mean_squared_error(y_test, p), explained_variance_score(y_test, p),
            len(m[-1].support_)) + "".join("{:>9.2f}".format(v) for v in preds))

print("\nTarget (medv, thousand $): mean = {:.2f}, std = {:.2f}".format(y.mean(), y.std()))
