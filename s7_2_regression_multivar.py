"""Раздел 7.2. Многомерный регрессор: линейный и полиномиальный."""
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)
print("\nPolynomial features:", X_train_transformed.shape[1])

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

print("\nDatapoint:", datapoint[0], "| nearest row in file: [7.66, 6.29, 5.66] -> 41.35")
print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

print("\nTest-set comparison by polynomial degree:")
print("{:>6}{:>10}{:>10}{:>10}{:>12}".format("degree", "n_feat", "MAE", "R2", "pred(dp)"))
for d in (1, 2, 3, 5, 10):
    pf = PolynomialFeatures(degree=d)
    m = linear_model.LinearRegression().fit(pf.fit_transform(X_train), y_train)
    p = m.predict(pf.transform(X_test))
    print("{:>6}{:>10}{:>10.2f}{:>10.3f}{:>12.2f}".format(
        d, pf.n_output_features_, sm.mean_absolute_error(y_test, p),
        sm.r2_score(y_test, p), m.predict(pf.transform(datapoint))[0]))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
print("\nDegree 10 + StandardScaler + Ridge:")
for alpha in (0.1, 1.0, 10.0):
    m = make_pipeline(StandardScaler(), PolynomialFeatures(degree=10), StandardScaler(),
                      linear_model.Ridge(alpha=alpha)).fit(X_train, y_train)
    p = m.predict(X_test)
    print("  alpha={:<5} MAE={:.2f}  R2={:.3f}  pred(dp)={:.2f}".format(
        alpha, sm.mean_absolute_error(y_test, p), sm.r2_score(y_test, p), m.predict(datapoint)[0]))
