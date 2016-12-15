import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt


f = open("train_data_f_pro.csv")
f2 = open("test_data_f_pro.csv")
data = np.loadtxt(f,delimiter=",")
data2 = np.loadtxt(f2,delimiter=",")

X = data[:,0:5]
X_two = data[:,3:5]
y = data[:,5]   
x2 = data2[:,0:5]
x2_two = data2[:,3:5]

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_2.fit(X, y)
y_pred = regr_2.predict(x2)
err_mse = mean_squared_error(data2[:,5], y_pred)
err_mean = mean_absolute_error(data2[:,5], y_pred)
r2 = r2_score(data2[:,5], y_pred)

regr_3 = DecisionTreeRegressor(max_depth=5)
regr_3.fit(X_two,y)
y_pred2 = regr_3.predict(x2_two)
err_mse2 = mean_squared_error(data2[:,5], y_pred2)
err_mean2 = mean_absolute_error(data2[:,5], y_pred2)
r2_2 = r2_score(data2[:,5], y_pred2)


print("All features")
print y_pred
print ("All features: mse_error", err_mse)
print("All features: rms_error:", (err_mse**0.5))

print("All features: r2:", r2)
print("All features: mean_error:", err_mean)

print("Two features")
print y_pred2
print ("two features: mse_error", err_mse2)
print("two features: rms_error:", (err_mse2**0.5))

print("two features: r2:", r2_2)
print("two features: mean_error:", err_mean2)

plt.figure(1)
plt.plot(data2[:,5],data2[:,5],linestyle = "-", label ="data")
plt.plot(data2[:,5], y_pred, linestyle ="-", label="Decision Tree Regression (5 features)")
plt.legend(loc='upper right',prop={'size':6})
plt.xlim(5,10)
plt.ylim(1,10)
plt.title('Movie Prediction')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')

plt.figure(2)
plt.plot(data2[:,5],data2[:,5],linestyle = "-", label ="data")
plt.plot(data2[:,5], y_pred2, linestyle ="-", label="Decision Tree Regression (2 features)")
plt.legend(loc='upper right',prop={'size':6})
plt.xlim(5,10)
plt.ylim(1,10)
plt.title('Movie Prediction')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')

plt.show()
