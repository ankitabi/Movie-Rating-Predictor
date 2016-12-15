import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

f = open("D:\\USC_Aditya\\Internet&Cloud-EE542\\Project\\twitter code\\newtest\\input2.csv")
f2 = open("D:\\USC_Aditya\\Internet&Cloud-EE542\\Project\\twitter code\\newtest\\test_input2.csv")

data = np.loadtxt(f,delimiter=",")
data2 = np.loadtxt(f2,delimiter=",")
X = data[:, 0:2] 
y = data[:, 2]  
x2 = data2[:,0:2]
test_res = data2[:,2]
pred = []
pred_svm = []
pred_svmpol = []
pred_regr = []
clf = linear_model.LinearRegression()
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
regr_1 = Ridge(alpha=1.0)
clf.fit(X, y)
svr_lin.fit(X, y)
svr_poly.fit(X, y)
regr_1.fit(X, y)
print clf.coef_
print clf.intercept_

print regr_1.coef_
print regr_1.intercept_
i = 0;
for val in clf.predict(x2):
    print str(x2[i]) + "  "+ str(data2[i][2]) + "  " + str(val)
    i += 1
    pred.append(val)
i=0
print "\n\n"
for val in svr_lin.predict(x2):
    print str(x2[i]) + "  "+ str(data2[i][2]) + "  " + str(val)
    i += 1
    pred_svm.append(val)
i=0;
print "\n\n"
for val in svr_poly.predict(x2):
    print str(x2[i]) + "  "+ str(data2[i][2]) + "  " + str(val)
    i += 1
    pred_svmpol.append(val)
i=0
print "\n\n"
for val in regr_1.predict(x2):
    print str(x2[i]) + "  "+ str(data2[i][2]) + "  " + str(val)
    i += 1
    pred_regr.append(val)

print "RMSE values"
print np.sqrt(mean_squared_error(np.array(test_res), np.array(pred)))
print np.sqrt(mean_squared_error(np.array(test_res), np.array(pred_svm)))
print np.sqrt(mean_squared_error(np.array(test_res), np.array(pred_svmpol)))
print np.sqrt(mean_squared_error(np.array(test_res), np.array(pred_regr)))

print "R2"
print clf.score(x2, test_res)
print svr_lin.score(x2, test_res)
print svr_poly.score(x2, test_res)
print regr_1.score(x2, test_res)

print "MAE"
print mean_absolute_error(np.array(test_res), np.array(pred))
print mean_absolute_error(np.array(test_res), np.array(pred_svm))
print mean_absolute_error(np.array(test_res), np.array(pred_svmpol))
print mean_absolute_error(np.array(test_res), np.array(pred_regr))

print test_res
print pred
print pred_svm
print pred_svmpol
print pred_regr

lw = 2
plt.gca().set_xlim([5,10]);
plt.gca().set_ylim([1,10]);
plt.rc('font',size=20)
plt.rc('legend',fontsize=15)
plt.plot(test_res, test_res, color='green',lw=lw, label='data')
plt.hold('on')
plt.plot(test_res, pred, color='navy', lw=lw, label='Linear Regression')
plt.plot(test_res, pred_svm, color='black', lw=lw, label='Linear SVR model')
plt.plot(test_res, pred_svmpol, color='red', lw=lw, label='Polynomial SVR model')
plt.plot(test_res, pred_regr, color='violet', lw=lw, label='Ridge Regression model')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Movie Prediction')
plt.legend()
plt.show()
