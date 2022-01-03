from data_health_insurance import read_data
import linear_regression as lr
import sys

# 1) Get data
(x, y, data) = read_data()
#sys.exit()

# 2) Train and predict
linReg = lr.linear_regression(x, y, normalize=True, normalize_columns=[0, 2, 3], alpha=0.03, iterations=30000, cv_size=0.2, test_size=0.0)
linReg.gradient_descent()

predict_test = linReg.predict(x[[1330], :], data_with_bias=False, data_normalized=False)
print("Predict data 1: ", x[[1330], :])
print("Predict result 1: ", predict_test)
print("Real result 1: ", y[1330])


