import pickletools
from pyexpat import model
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/kekli/Desktop/ML event/maas.csv")

veri=data.copy()

#print(veri.isnull().sum()) eksik datamız var mı? 

y=veri["Salary"]
X=veri["YearsExperience"]

#plt.scatter(X,y) verileri grafikte gördük...
#plt.show()

sabit=sm.add_constant(X)
model=sm.OLS(y,sabit).fit() #Ols Ekranını bastırdık
print(model.summary())

lr=LinearRegression()
lr.fit(X.values.reshape(-1,1),y.values.reshape(-1,1))
#.values diyerek array şekline getirdik.reshape diyerek sütün bazlı hale getircez

print(lr.coef_,lr.intercept_) #b1 ve sabitleri çağırdık
print(lr.predict(X.values.reshape(-1,1))) #tahmin idealleri
#modelimizi kurduk elimizdeki bağımsız değişkenleri koyarak tahmin 
#değerleri ortaya çıkartıyoruz.