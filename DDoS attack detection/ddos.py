'''
                                            T.C. Sakarya Üniversitesi Bilgisayar ve Bilişim Sistemleri Fakültesi
                                                            Bilgisayar Mühendsiliği Bölümü

                                                            Ağ Güvenliği Dersi Proje Ödevi
                                                    Ödev Konusu: DDoS Atak Tespiti ve Analizi

                                                            Deniz Berfin Taştan / B181210010 / 1-A
                                                        Mustafa Melih Tüfekcioğlu / B191210004 / 1-A

'''

#Gerekli kütüphanelerin eklenmesi
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import matplotlib.gridspec as gridspec 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import tree
import matplotlib.pyplot as plt


#Veri setinin projeye eklenmesi
file_path_20_percent = 'archive/KDDTrain_20Percent.txt' #NSL-KDD train setinin %20'lik alt kümesi
file_path_full_training_set = 'archive/KDDTrain.txt' #NSL-KDD 'nin train setinin tamamı
file_path_test = 'archive/KDDTest.txt'  #NSL-KDD' nin test setinin tamamı
df = pd.read_csv(file_path_full_training_set)
test_df = pd.read_csv(file_path_test)


#Txt dosyasi icinde basliklar bulunmadigi icin kendimiz ekledik.
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level']) 

df.columns = columns
test_df.columns = columns

df.describe() #Veri setindeki özniteliklerin min,max,standart sapma gibi değerlerini görüntülenmesi.

df.info() #Verideki özniteliklerin veri tiplerini ve null değer olup olmadığını kontrol edilmesi.
test_df.info()

df.nunique()
test_df.nunique() 



#Boş veri sayısını çıkarttık.
pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/len(df)], 
              axis=1).rename(columns={0:'Missing Records', 1:'Percentage (%)'}) 



#Histogram diyagramı çıkartmak için fonksiyon tanımladık.
def plot_hist(df, cols, title):
    grid = gridspec.GridSpec(10, 2, wspace=0.5, hspace=0.5) 
    fig = plt.figure(figsize=(15,25)) 
    
    for n, col in enumerate(df[cols]):         
        ax = plt.subplot(grid[n]) 

        ax.hist(df[col], bins=20) 
        ax.set_title(f'{col} distribution', fontsize=15) 
    
    fig.suptitle(title, fontsize=20)
    grid.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()



#Özniteliklerin ve dağılımlarının histogram diyagramlarını oluşturduk.
hist_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
plot_hist(df, hist_cols, 'Integer Olan Öznitelikler')
rate_cols = [ 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
plot_hist(df, rate_cols, 'Oransal Olan Öznitelikler')



#verileri atak olan ve olmayan olarak ayırdık.
is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)



#veri setine atak durumunu gösteren bir kolon ekledik.
df['attack_state'] = is_attack
test_df['attack_state'] = test_attack

(df.attack_state == 1).sum()/len(df) #Train veri setimizde yaklaşık %46 atağımız var.
(test_df.attack_state == 1).sum()/len(df) #Test veri setimizde yaklaşık %10 atağımız var.

#Atak durumunun yoğunluğunu grafiğe döken fonksiyon
sns.kdeplot(
   data=df, x="attack_state",
   fill=True, common_norm=False, palette="crest",
   alpha=.2, linewidth=10,
)

#Atak olan durumlar 1, olmayan durumlar 0 olarak işaretlendi.
attack = (df.attack_state == 1).sum()
nonAttack = (df.attack_state == 0).sum()
myData = [attack , nonAttack]

#Atak durumu yüzdesi pasta grafiğine döküldü.
my_labels = 'Atak Sayısı','Atak Olmayan Durum Sayısı'
plt.pie(myData,labels=my_labels,autopct='%1.1f%%')
plt.title('Atak Oranı')
plt.axis('equal')
plt.show()

#Atakların, alt atak tiplerini içerdiği sınıflar oluşturduk.
DoS_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
Probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
U2R = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
R2L = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

attack_labels = ['Normal','DoS','Probe','U2R','R2L']

#Atakları sınıflandırdık.
def class_attack(attack):
    if attack in DoS_attacks:
        attack_type = 1
    elif attack in Probe_attacks:
        attack_type = 2
    elif attack in U2R:
        attack_type = 3
    elif attack in R2L:
        attack_type = 4
    else:
        attack_type = 0       
    return attack_type

#Bu sütunun altında ataklarımızın sınıfların sayılsal karşılıkları bulunur.
attack_class = df.attack.apply(class_attack)
df['attack_class'] = attack_class

test_attack_class = test_df.attack.apply(class_attack)
test_df['attack_class'] = test_attack_class

df.head()

#Atak sınıfları yüzdesel olarak hesaplandı.
Normal = (df.attack_class == 0).sum()/len(df)
print('Normal = ' , Normal)
DoSDDoS = (df.attack_class == 1).sum()/len(df)
print('DoS/DDoS = ' , DoSDDoS)
Probe = (df.attack_class == 2).sum()/len(df)
print("Probe = " , Probe )
U2R = (df.attack_class == 3).sum()/len(df)
print('U2R = ', U2R)
R2L = (df.attack_class == 4).sum()/len(df)
print('R2L = ' ,R2L)

Normal = (test_df.attack_class == 0).sum()/len(test_df)
print('Normal = ' , Normal)
DoSDDoS = (test_df.attack_class == 1).sum()/len(test_df)
print('DoS/DDoS = ' , DoSDDoS)
Probe = (test_df.attack_class == 2).sum()/len(test_df)
print("Probe = " , Probe )
U2R = (test_df.attack_class == 3).sum()/len(test_df)
print('U2R = ', U2R)
R2L = (test_df.attack_class == 4).sum()/len(test_df)
print('R2L = ' ,R2L)

attack_vs_class = pd.crosstab(df.attack_class, df.attack)
attack_vs_class

attack_vs_DDoS = pd.crosstab(df.attack_class == 1, df.attack)
attack_vs_DDoS


#pasta grafiği çizen fonksiyon
def bake_pies(data_list,labels):
    list_length = len(data_list)
    
    color_list = sns.color_palette()
    color_cycle = itertools.cycle(color_list)
    cdict = {}

    fig, axs = plt.subplots(1, list_length,figsize=(18,10), tight_layout=False)
    plt.subplots_adjust(wspace=1/list_length)

    for count, data_set in enumerate(data_list): 
        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)
        wedges,texts = axs[count].pie(data_set,
                           colors=[cdict[v] for v in data_set.index])
        axs[count].legend(wedges, data_set.index,
                           title="Durum",
                           loc="center left",
                           bbox_to_anchor=(1, 0, 0.5, 1))
        axs[count].set_title(labels[count])
    return axs 


#Atak alt sınıfları yüzdelendirildi ve grafiğe döküldü.
DoSDDoS_class = df.loc[df.attack_class == 1].attack.value_counts()
probe_class = df.loc[df.attack_class == 2].attack.value_counts()
flag_axs = bake_pies([DoSDDoS_class , probe_class], ['DoS/DDoS','Probe'])        
plt.show()
U2R_class = df.loc[df.attack_class == 3].attack.value_counts()
R2L_class = df.loc[df.attack_class == 4].attack.value_counts()
flag_axs = bake_pies([U2R_class,R2L_class], ['U2R','R2L'])        
plt.show()

Normal = (df.attack_class == 0).sum()
DoSDDoS = (df.attack_class == 1).sum()
Probe = (df.attack_class == 2).sum()
U2R = (df.attack_class == 3).sum()
R2L = (df.attack_class == 4).sum()
myData = [Normal , DoSDDoS,Probe,U2R,R2L]

my_labels = 'Normal','DoS/DDoS' ,'Probe' ,'U2R' , 'R2L'
plt.pie(myData,labels=my_labels ,autopct='%1.1f%%' , shadow = True)
plt.title('Atak Sınıfları\n\n')
plt.axis('equal')
plt.show()

#DDoS atakları protokol ve servis özellikleri ile ayrıldı ve grafik üzerinde gösterildi.
attack_vs_protocol = pd.crosstab((df.attack_class == 1), df.protocol_type)
attack_vs_protocol
icmp = attack_vs_protocol.icmp.sum()
tcp = attack_vs_protocol.tcp.sum()
udp = attack_vs_protocol.udp.sum()
myData = [icmp , tcp,udp]
my_labels = 'icmp','tcp' ,'udp' 
plt.pie(myData,labels=my_labels ,autopct='%1.1f%%' , shadow = True)
plt.title('protocol type\n\n')
plt.axis('equal')
plt.show()

icmp_attacks = attack_vs_protocol.icmp
tcp_attacks = attack_vs_protocol.tcp
udp_attacks = attack_vs_protocol.udp


bake_pies([icmp_attacks, tcp_attacks, udp_attacks],['icmp','tcp','udp'])
plt.show()


normal_services = df.loc[df.attack_class == 0].service.value_counts()
DDoS_attack_services = df.loc[df.attack_class == 1].service.value_counts()


service_axs = bake_pies([normal_services, DDoS_attack_services], ['normal','DDoS_attack'])        
plt.show()

service_vs_protocol = pd.crosstab(df.service, df.protocol_type,)
service_vs_protocol

icmp_service = service_vs_protocol.icmp
tcp_service = service_vs_protocol.tcp
udp_service = service_vs_protocol.udp

bake_pies([icmp_service, tcp_service, udp_service],['icmp','tcp','udp'])
plt.show()

df = pd.get_dummies(df,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")
test_df = pd.get_dummies(test_df,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

drop_cols = ['attack' ]  
df.drop(drop_cols, axis=1, inplace=True)    

drop_cols = ['attack' ]  
test_df.drop(drop_cols, axis=1, inplace=True)  
normal = df[df.attack_class == 0]

normal_test= test_df[test_df.attack_class == 0]
DDoS = df[df.attack_class == 1]
DDoS_test= test_df[test_df.attack_class == 1 ]
total_data = pd.concat([normal, DDoS], ignore_index=True)
total_data_test = pd.concat([normal_test, DDoS_test], ignore_index=True)

#Bağımlılıkları ölçmek için korelasyon katsayısı hesapladık.
corr= total_data.corr()
corr_y = abs(corr['attack_class'])
highest_corr = corr_y[corr_y > 0.1]
highest_corr.sort_values(ascending=True)

corr= total_data_test.corr()
corr_y = abs(corr['attack_class'])
highest_corr_test = corr_y[corr_y >0.1]
highest_corr_test.sort_values(ascending=True)

#Hesaplanan korelasyon katsayısına göre ısı haritası oluşturduk.
highest_corr_columns= highest_corr.index
highest_corr_test_columns= highest_corr_test.index
plt.figure(figsize=(15,10))
g=sns.heatmap(total_data[highest_corr.index].corr(),annot=True,cmap="RdYlGn")

plt.figure(figsize=(15,10))
g=sns.heatmap(total_data_test[highest_corr_test.index].corr(),annot=True,cmap="RdYlGn")

#Oluşan ısı haritasına göre dataset'in kolonları yeniden yapılandırdık.
drop_cols = df.loc[:,[i for i in list(df.columns) if i not in [
'diff_srv_rate',                
'dst_host_same_src_port_rate', 
'REJ',
'tcp',                            
'ecr_i',                         
'rerror_rate',                    
'srv_rerror_rate',                
'dst_host_srv_rerror_rate',       
'dst_host_rerror_rate',           
'smtp',                           
'dst_host_srv_diff_host_rate',    
'domain_u',                       
'udp',                            
'srv_diff_host_rate',             
'private',                        
'dst_host_count',                 
'http',                           
'logged_in',                      
'count',
'dst_host_srv_count',             
'dst_host_same_srv_rate',         
'serror_rate',                    
'srv_serror_rate',                
'dst_host_serror_rate',           
'S0',                             
'dst_host_srv_serror_rate',       
'SF',                             
'same_srv_rate',                 
'attack_state',                  
'attack_class', 
'other',
'icmp',                         
'wrong_fragment',               
'dst_host_diff_srv_rate',   
'RSTO',
'ftp_data',
'Z39_50',
'uucp'                          
]]]

df.drop(drop_cols, axis=1, inplace=True)  
df 

drop_cols1 = test_df.loc[:,[i for i in list(test_df.columns) if i not in [
'diff_srv_rate',                
'dst_host_same_src_port_rate', 
'REJ',
'tcp',                            
'ecr_i',                         
'rerror_rate',                    
'srv_rerror_rate',                
'dst_host_srv_rerror_rate',       
'dst_host_rerror_rate',           
'smtp',                           
'dst_host_srv_diff_host_rate',    
'domain_u',                       
'udp',                            
'srv_diff_host_rate',             
'private',                        
'dst_host_count',                 
'http',                           
'logged_in',                      
'count',                          
'dst_host_srv_count',             
'dst_host_same_srv_rate',         
'serror_rate',                    
'srv_serror_rate',                
'dst_host_serror_rate',           
'S0',                             
'dst_host_srv_serror_rate',       
'SF',                             
'same_srv_rate',                 
'attack_state',                 
'attack_class', 
'other',
'icmp',                         
'wrong_fragment',               
'dst_host_diff_srv_rate',   
'RSTO',
'ftp_data',
'Z39_50',
'uucp'
]]]

test_df.drop(drop_cols1, axis=1, inplace=True)  
test_df

data = df.copy()
test_data = test_df.copy()
X_train = df.drop('attack_class'  , axis = 1)
X_test = test_df.drop('attack_class' , axis = 1)
y_train = df['attack_class']
y_test = test_df['attack_class']

#Tahmin işleminden sonra bu tahminler ile karmaşıklık matrisi oluşturuan fonksiyon.
def add_predictions(data_set,predictions,y):
    prediction_series = pd.Series(predictions, index=y.index)

    predicted_vs_actual = data_set.assign(predicted=prediction_series)
    original_data = predicted_vs_actual.assign(actual=y).dropna()
    conf_matrix = confusion_matrix(original_data['actual'], 
                                   original_data['predicted'])
    
    base_errors = original_data[original_data['actual'] != original_data['predicted']]
    
    non_zeros = base_errors.loc[:,(base_errors != 0).any(axis=0)]

    false_positives = non_zeros.loc[non_zeros.actual==0]
    false_negatives = non_zeros.loc[non_zeros.actual==1]

    prediction_data = {'data': original_data,
                       'confusion_matrix': conf_matrix,
                       'errors': base_errors,
                       'non_zeros': non_zeros,
                       'false_positives': false_positives,
                       'false_negatives': false_negatives}
    
    return prediction_data

#Ölçekleme
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train) 
X_test= mms.transform(X_test)

#Naive Bayes Algoritması
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred = gnb.predict(X_test)
print("GNB Accuracy : ",metrics.accuracy_score(y_test,gnb_pred))

#Karar Ağaçları (Decision Tree) Algoritması
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
dt_pred = clf.predict(X_test)
print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, dt_pred))

#K-Neigbors Algoritması
knn = KNeighborsClassifier(n_neighbors = 6)
knn = knn.fit(X_train , y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:",metrics.accuracy_score(y_test, knn_pred))

#Random Forest Algoritması
rm = RandomForestClassifier()
rm.fit(X_train,y_train)
rm_pred=rm.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, rm_pred))

#Support Vector Machines
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:",metrics.accuracy_score(y_test, svm_pred))


