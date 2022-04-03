# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:02:01 2020

@author: busra
"""
#Kütüphaneler
import pandas as pd 
#veri yükleme
veriler = pd.read_csv("bank.csv")
print(veriler)
#veri keşif
eksikveri= veriler.isnull().sum()
print(eksikveri)
#istatistik tablosu
istatistikler = veriler.describe()
print(istatistikler)
print(veriler['y'].value_counts())
import seaborn as sn
import matplotlib.pyplot as plt
sn.countplot(x='y', data=veriler, palette='husl')
plt.show()
#korelasyon matrisi (keşif devam)
import seaborn as sn
import matplotlib.pyplot as plt
print("Korelasyon matrix'i: ")
corrmatrix = veriler.corr()
plt.figure(figsize=(10,7))
sn.heatmap(corrmatrix, annot=True, )
plt.show()
#ordinal encoder kullanımı (ver ön işleme)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
oe.fit(veriler[["job","marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]])
veriler[["job","marital", "education", "default", "housing", 
         "loan", "contact", "month", "poutcome", "y"]] = oe.transform(veriler[["job","marital", "education", "default", "housing", 
                                                     "loan", "contact", "month", "poutcome", "y"]])
#girdi(x) ve çıktı(y) kolonlarının birbirinden ayrilması
x = veriler.iloc[:, :-1]
y = veriler.iloc[:, 16:17]
#eğitim ve test verisinin ayrıştırılması
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state=0)
#Multiple regression (Çoklu regresyon)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train, y_train)
#tahmin sonuçları
y_tahmin = model1.predict(x_test)
# modelin başarı scorlaması r2
from sklearn.metrics import r2_score
model1_basari = r2_score(y_test, y_tahmin)
print(model1_basari)
#Feature selection (Öznitelik seçme)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
# feature extraction
X = veriler.iloc[:, :-1]
Y = veriler.iloc[:, 16:17]
test = SelectKBest(score_func=f_regression, k=5)
fit = test.fit(X, Y)
# summarize scores
print(fit.scores_)
features = fit.transform(X)
#öznitelikleri modele ekleme
ftr = pd.DataFrame(data=features, columns=["housing", "contact", "duration", "p-day", "previous"])
X_train, X_test, Y_train, Y_test = train_test_split(ftr, Y, test_size =0.3, random_state=5)
#verileri normalize ederek skor sonucunu iyileştirmeye çalışma
from sklearn import preprocessing
normalized_X = preprocessing.normalize(ftr)
normalized_Y = preprocessing.normalize(Y)
X_train, X_test, Y_train, Y_test = train_test_split(normalized_X, normalized_Y, test_size =0.3, random_state=5)
#model2 regresyon
model2 = LinearRegression()
model2.fit(X_train, Y_train)
# model2 tahmin sonuçları 
Y_normalized_tahmin = model2.predict(X_test)
# model2 r2 score
model2_basari = r2_score(Y_test, Y_normalized_tahmin)
print(model2_basari)
#normalizasyonun başarısız olmasından sonra kaldırılarak seçilen özniteliklerle tekrar bir regresyon modelinin oluşturulması 
X_train, X_test, Y_train, Y_test = train_test_split(ftr, Y, test_size =0.3, random_state=5)
#model2 regresyon
model = LinearRegression()
model.fit(X_train, Y_train)
# model2 tahmin sonuçları 
Y_tahmin = model.predict(X_test)
# model2 r2 score
model_basari = r2_score(Y_test, Y_tahmin)
print(model_basari)
#lojistik regresyon modeli
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# log_reg tahmin sonuçları
Y_log_pred = logreg.predict(X_test)
# Tahmin doğruluk oranının ekrana yazdırılması
log_score = logreg.score(X_test, Y_test)
print('test veri setinin tahmin doğruluk oranı:', log_score)
# confusion matrix görünümü ve ekrana bastırılması
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_log_pred)
print('Confusion matrix:')
print(confusion_matrix)