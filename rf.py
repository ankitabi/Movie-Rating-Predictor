from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
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

rf = RandomForestRegressor(n_estimators=12, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False)
rf.fit(X,y)
y_pred = rf.predict(x2)
err_mse = mean_squared_error(data2[:,5], y_pred)
err_mean = mean_absolute_error(data2[:,5], y_pred)
r2 = r2_score(data2[:,5], y_pred)


rf.fit(X_two,y)
y_pred2 = rf.predict(x2_two)
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
plt.plot(data2[:,5],data2[:,5],color = 'g',linestyle = "-", label ="data")
plt.plot(data2[:,5], y_pred,color = 'b' ,linestyle ="-", label="Random forest Regression (5 features)")
plt.legend(loc='upper right',prop={'size':6})
plt.xlim(5,10)
plt.ylim(1,10)
plt.title('Movie Prediction')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')


plt.figure(2)
plt.plot(data2[:,5],data2[:,5],color = 'g',linestyle = "-", label ="data")
plt.plot(data2[:,5], y_pred2, linestyle ="-", color = 'b',label="Random forest Regression (2 features)")
plt.legend(loc='upper right',prop={'size':6})
plt.xlim(5,10)
plt.ylim(1,10)
plt.title('Movie Prediction')
plt.xlabel('Predicted Rating')
plt.ylabel('Actual Rating')

plt.show()
