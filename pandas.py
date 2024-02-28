import pandas as pd 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

veri=pd.read_csv("C:/Users/Kerem/Desktop/veriseti_20220203_olimpiyatlar.csv")
print(veri.head()) #ilk 5 elemanı görme
print(veri.info()) #veri seti hakkında bilgi alma
print(veri.columns) #sütün isimlerini getirme

#sutun isimlerini değiştirme
veri.rename(columns={
                     'ID' : 'id',
                     'isim':'Name',
                     'cinsiyet': 'Gender',
                     'yas':'Age' ,
                     'boy': 'Height',
                     'kilo': 'Weight',
                     'takim':'Team',
                     'uok':'NOC',  
                     'yil': 'Year',
                     'sezon': 'Season',
                     'spor':'sport',
                     'sehir': 'City',
                     'etkinlik':'Event',
                     'madalya':'Medal'}, inplace = True)
print(veri.head())
print(veri.columns)

#yararsız verilerin çıkarılması ve sorunun düzeltilmesi

veri=veri.drop(["id"],axis=1) #axis=1 sütün manasındadır
print(veri.head(2))


#kayıp veri sorunu:
yas_ortalama=np.round(np.mean(veri.Age),2)
print("yas ortalaması : {}".format(yas_ortalama))
veri["Age"]=veri["Age"].fillna(yas_ortalama) #eksik yaş değerlerini yaş sütununun ortalaması ile doldurarak veri çerçevesindeki eksik verileri ele alır.
print(veri.info())




#madalya alamayan sporcuları veri setinden çıkar
madalya_degisken=veri["Medal"]
print(pd.isnull(madalya_degisken).sum()) #madalya alamayanların sayisini
madalya_degisken_filtresi=~pd.isnull(madalya_degisken) #dolu olnaları getirdik ve filtre içine koyduk
veri=veri[madalya_degisken_filtresi] # madalya almış olanları ekledim geriye kalmış olanları düşürdük
print(veri.head(5))

#sonradan kullanmak için veriyi kaydedicez
veri.to_csv("olimpiyatlar_temizlenmiş.csv",index=False)


#TEK DEĞİŞKENLİ VERİ ANALİZİ
#A-)Sayısal değişkenler
#1-)Histogram grafiğini çizdiricek fonksiyon:

def plothistogram(degisken):
    #girdi:değişken/sutun ismi    çıktı:ilgili değişkenin histogramı
    plt.figure()
    plt.hist(veri[degisken],bins=85,color="orange")
    plt.xlabel(degisken)
    plt.ylabel("frekans")
    plt.title("Veri sıklığı - {}".format(degisken))
    plt.show()

#tüm sayısal değişkenler için histogramları çizdirerim:
sayısal_degisken=["yas","boy","kilo","yil"]
for i in sayısal_degisken:
    plothistogram(i)


#sayısal verinin istatistiksel  özelikleri:
print(veri.describe())

#2-)Kutu grafiği:
plt.boxplot(veri.yas)
plt.title("yas degiskeni için kutu grafiği")
plt.xlabel("yas")
plt.ylabel("deger")
plt.show()

#B-)Kategrik degiskenler:
def plotbar(degisken,n=5):
    verimiz=veri[degisken]
    veri_sayma=verimiz.value_counts() #veri_sayma = verimiz.value_counts(): Bu satır, verimiz Serisi içindeki benzersiz değerlerin frekanslarını hesaplar ve sonuçları veri_sayma adlı bir Seri olarak saklar.
   
    veri_sayma=veri_sayma[:n] #ilk beş tanesini istedik en çok olan 5 tanesini. sadece en yüksek frekansa sahip olan 5 veri görüntülenecektir.
    plt.figure()
    plt.bar(veri_sayma.index, veri_sayma, color = "orange")  #bar( çubukların x ekseni değerlerini, çubukların y ekseni değerlerini)
    plt.xticks(veri_sayma.index, veri_sayma.index.values) # 1. deger veri_sayma adlı bir Pandas Serisi'nin indeks değerlerini içerir,2. deger ise  x ekseni etiketlerinin ne olarak görüntüleneceğini belirler. Örneğin, "İstanbul" ve "Ankara" gibi şehir isimlerini içerir.
    plt.xticks(rotation=45) #
    plt.ylabel("Frekans")
    plt.title("Veri Sakliga - {}".format (degisken))
    plt.show()
    print("{} : \n {}".format(degisken, veri_sayma))#konsola yazar .Bu satır, çizilen sütunun adını ve bu sütundaki benzersiz değerlerin frekanslarını ekrana yazdırmak için kullanılır

kategorik_degisken=["isim", "cinsiyet", "takim","uok", "sezon","sehir", "spor","etkinlik","madalya"]
for i in kategorik_degisken:
    plotbar(i)


#İKİ DEĞİŞKENLİ VERİ ANALİZİ
#1-)Cinsiyete göre ve ağırlık karşılaştırması:
erkek=veri[veri.cinsiyet == "M"]
#print(erkek.head(3))

kadın=veri[veri.cinsiyet == "F"]
#print(kadın.head(3))

plt.figure()
plt.scatter(kadın.boy,kadın.kilo,alpha=0.4,label="kadın",color="orange") #saçılım dağılım grafiği çizicez kolersayon varmı görmeye çalışıcaz
plt.scatter(erkek.boy,erkek.kilo,alpha=0.4,label="erkek",color="blue") 
plt.xlabel("boy")
plt.ylabel("kilo")
plt.title("boy kilo ilişki")
plt.legend()
plt.show()

#sayısal sutunlar arası ilişki inceleme:
 #kolerasyon tablosu yapar
print(veri.loc[:,["yas","boy","kilo"]].corr())


#madalya yas ilişkisi:
veri_gecici=veri.copy()
veri_gecici=pd.get_dummies(veri_gecici,columns=['madalya'])#madalya altın bronz gümüş stunları oluşturdu her satırda kim ne kazandıysa ona true  diğerlerine false  yazdı
print(veri_gecici.head(2))
print(veri_gecici.loc[:,["yas","madalya_Bronze","madalya_Gold","madalya_Silver"]].corr())

#takımların kazandıkları madalyalar
print(veri_gecici[["takim","madalya_Bronze","madalya_Gold","madalya_Silver"]].groupby("takim",as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)[:5]) #takim bazında grupla degerleri topla ve sonrada altın madalya sayısına göre sırala ve yukardan aşağı olsun ilk 5 tane gör

#kazanılan madalyaların  hangi şehirde kazanıldıgı:
print(veri_gecici[["sehir","madalya_Bronze","madalya_Gold","madalya_Silver"]].groupby("sehir",as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)[:5]) 

#cinsiyete göre altın bronz gümüş madalya sayıları:
print(veri_gecici[["cinsiyet","madalya_Bronze","madalya_Gold","madalya_Silver"]].groupby("cinsiyet",as_index=False).sum().sort_values(by="madalya_Gold",ascending=False)) 



#ÇOL DEĞİŞKENLİ VERİ ANALİZİ
veri_pivot = veri.pivot_table(index="madalya", columns="cinsiyet",
                              values= ["boy", "kilo", "yas"],
                              aggfunc={"boy" : np.mean, "kilo": np.mean, "yas":[min,max,np.std]})
print(veri_pivot.head())

#anomali test:
def anomaliTespiti(df, ozellik):
    outlier_indices = [] #aykırı değerleri bulup bunun içine yazacaz
    for c in ozellik:
        # 1. ceyrek
        Q1 = np.percentile(df[c], 25)
        # 3. ceyrek
        Q3 = np.percentile(df [c], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    kerem =list(i for i, v in outlier_indices.items() if v > 1)
  
    return kerem

veri_anomali=veri.loc[anomaliTespiti(veri,["yas", "kilo", "boy"])]
print(veri_anomali.spor.value_counts())



#ZAMAN SERİSİNDE VERİ ANALİZİ

veri_zaman=veri.copy()
print(veri_zaman.head(2))

#Olimpiyatlar gerçekleştiği yıllar
essiz_yıllar=veri_zaman.yil.unique()
print(essiz_yıllar)

#olimpiyatları yapıldıgı yıllara göre sırala:
dizili_array=np.sort(veri_zaman.yil.unique())
print(dizili_array)

plt.figure()
plt.scatter(range(len(dizili_array)), dizili_array)
plt.grid(True)
plt.ylabel("Villar")
plt.title("olimpiyatlar Çift Villarda Düzenlenir")
plt.show()



























