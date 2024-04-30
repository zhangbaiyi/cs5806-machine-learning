import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

np.random.seed(5806)


# Consider an ARMA(2,2) process
# y(t) - 0.5y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2)
# true mean = 0
# true var = 3
arparams = np.array([0.5, 0.2])
maparams = np.array([0.1, 0.4])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag

T = 10000
var = 1
arma_process = sm.tsa.ArmaProcess(ar,ma)
y = arma_process.generate_sample(T, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) - 0.5y(t-1) - 0.2y(t-2) = e(t) + 0.1e(t-1) + 0.4e(t-2)")
plt.ylabel("Magnitude - y(t)")
plt.xlabel("Samples")
plt.show()

# Experimental mean & variance
print("The experimental mean of y(t) is: " + str(np.round(np.mean(y), 2)))
print("The experimental var of y(t) is: " + str(np.round(np.var(y), 2)))

# #
# # The experimental mean of y(t) is: 0.09
# # The experimental var of y(t) is: 2.93

# Q2
def q2_prompt():
    i_samples = input("a. Enter the number of data samples: ___")
    i_var = input("b. Enter the variance of the white noise: ___")
    i_ar_order = input("c. Enter AR order: ___")
    i_ma_order = input("d. Enter MA order: ___")
    i_co_ar = input("e. Enter the coefficients of AR: ___ (Use space to split)")
    i_co_ma = input("f. Enter the coefficients of MA: ___ (Use space to split)")
    i_co_ar_num = []
    i_co_ma_num = []
    print("""
    Warning: if the number of coefficients you provided 
    is not equal to the corresponding order, 
    assert error will be thrown 
    """)
    try:
        i_samples = int(i_samples)
        i_var = int(i_var)
        i_ar_order = int(i_ar_order)
        i_ma_order = int(i_ma_order)
        i_co_ar_num = [float(num) for num in i_co_ar.split()]
        i_co_ma_num = [float(num) for num in i_co_ma.split()]
        assert len(i_co_ar_num) == i_ar_order
        assert len(i_co_ma_num) == i_ma_order
    except ValueError:
        print("Invalid input")

    _T = i_samples
    _var = i_var
    _ar = np.r_[1, i_co_ar_num]  # add zero-lag and negate
    _ma = np.r_[1, i_co_ma_num]  # add zero-lag
    print(_ar, _ma)
    _arma_process = sm.tsa.ArmaProcess(_ar, _ma)
    _y = arma_process.generate_sample(_T, scale=np.sqrt(_var))

    plt.plot(_y)
    plt.title("Custom function")
    plt.ylabel("Magnitude")
    plt.xlabel("Samples")
    plt.grid()
    plt.show()

    return i_samples, i_var, i_ar_order, i_ma_order, i_co_ar_num, i_co_ma_num


print(q2_prompt())
# samples, var, ar_order, ma_order, ar_co, ma_co = q2_prompt()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ACF_PACF_Plot(y, lags):
    _acf = sm.tsa.stattools.acf(y, nlags=lags)
    _pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()
    return _acf, _pacf

import seaborn as sns
def myGpac(Ry, na, nb):
    # LOOP: k FROM 1 to na
    #      LOOP: j from 0 to nb
    df = pd.DataFrame(np.zeros((nb, na)))
    print(df)

    def getRy(index):
        if index<0:
            return Ry[-index]
        else:
            return Ry[index]

    for k in range(1, na):
        for j in range(nb):
            if k == 1:
                denominator = Ry[j]
                nominator = Ry[j+k]
                if denominator == 0:
                    df[k][j] = np.nan
                else:
                    df[k][j] = nominator / denominator
            else:
                denominator = np.zeros((k,k))
                col = 0
                for i in range(j, j-k, -1):
                    for p in range(k):
                        denominator[p][col] = getRy(i+p)
                    col += 1
                # print(denominator)
                nominator = denominator.copy()
                for i in range(1, k+1):
                    nominator[i-1][k-1] = getRy(j+i)
                # print(nominator)
                det_nominator = np.linalg.det(nominator)
                det_denominator = np.linalg.det(denominator)
                print(det_nominator, det_denominator)
                if det_denominator == 0:
                    df[k][j] = np.inf
                else:
                    df[k][j] = det_nominator / det_denominator
    df.drop(0, axis=1, inplace=True)
    print(df)
    sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.3f')
    plt.title("GPAC Table")
    plt.show()


# Example 1: 𝑦(𝑡) − 0.5𝑦(𝑡 − 1) = 𝑒(𝑡)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 1, 0, [-0.5], []
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) - 0.5y(t-1) = e(t)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

acf, pacf = ACF_PACF_Plot(y, 20)

lags = 20
ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)

# Example 2: ARMA (0,1): y(t) = e(t) + 0.5e(t-1)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 0, 1, [], [0.5]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) = e(t) + 0.5e(t-1)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

acf, pacf = ACF_PACF_Plot(y, 20)

lags = 20
ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)

# Example 3: ARMA (1,1): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 1, 1, [0.5], [0.5]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)

# Example 4: ARMA (2,0): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 2, 0, [0.5, 0.2], []
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)

# Example 5: ARMA (2,1): y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 2, 1, [0.5, 0.2], [-0.5]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("y(t) + 0.5y(t-1) + 0.2y(t-2) = e(t) - 0.5e(t-1)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)


# Example 6: ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 1, 2, [0.5], [0.5, -0.4]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("Example 6: ARMA (1,2): y(t) + 0.5y(t-1) = e(t) + 0.5e(t-1) - 0.4e(t-2)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)


# Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 0, 2, [], [0.5, -0.4]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("Example 7: ARMA (0,2): y(t) = e(t) + 0.5e(t-1) - 0.4e(t-2)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)


# Example 8: ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)
np.random.seed(5806)
samples, var, ar_order, ma_order, ar_co, ma_co = 10000, 1, 2, 2, [0.5, 0.2], [0.5, -0.4]
ar = np.r_[1, ar_co]  # add zero-lag and negate
ma = np.r_[1, ma_co]  # add zero-lag

arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(samples, scale=np.sqrt(var))

plt.plot(y)
plt.title("Example 8: ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.grid()
plt.show()

lags = 20
acf, pacf = ACF_PACF_Plot(y, lags)

ry = arma_process.acf(lags=lags)
myGpac(ry, na=7, nb=7)


def plot_forecast(y):
    # Example 8: ARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)
    # yhat_{k}(1) = -0.6y(k-1) - 0.5yhat_{k-1}(1) + 0.4yhat_{k-2}(1)
    # k from 1 to 10001
    dp = [0] * (len(y) + 2)
    dp[0] = 0
    for _t in range(2, len(y)+1):
        dp[_t] = -0.6 * y[_t-1] - 0.5 * dp[_t-1] + 0.4 * dp[_t-2]
    y = np.append(y, ([[0],[0]]))
    y = pd.DataFrame(np.array(y).reshape(len(y), 1))
    dp = pd.DataFrame(np.array(dp).reshape(len(dp), 1))
    _df = pd.DataFrame(pd.concat([y, dp], axis=1))
    _df.columns = ['Actual', 'Forecast']
    plt.plot(_df['Actual'], label='y(t)')
    plt.plot(_df['Forecast'], label='ŷ(t)')
    plt.ylabel('Magnitude')
    plt.xlabel('Samples')
    plt.legend()
    plt.grid()
    plt.title("y(t) versus the one-step ahead prediction\nARMA (2,2): y(t)+0.5y(t-1) +0.2y(t-2) = e(t)+0.5e(t-1) - 0.4e(t-2)\nŷ{k}(1) = -0.6y(k-1) - 0.5ŷ{k-1}(1) + 0.4ŷ{k-2}(1)")
    plt.show()
    pass

plot_forecast(y)