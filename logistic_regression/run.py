from data_admissions import read_data
import logistic_regression as logr

(x, y, data) = read_data()

logr_admissions = logr.logistic_regression(x, y, normalize=True, normalize_columns=[0,1,2,3,4,5,6])
logr_admissions.train()