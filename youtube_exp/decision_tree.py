from quilt.data.haradai1262 import YouTuber
import collections
import numpy as np
import numpy.random as random
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)

def preprocessingDTree(pre_df):
    
    #タイトルの長さを確認
    pre_df['len_title'] = pre_df['title'].apply(lambda x: len(str(x).replace(' ', '')))
    
    #[]が含まれるなら1,含まれないなら0
    buzz_word_Braces = ["【","】",]
    for i in buzz_word_Braces:
        pre_df.loc[pre_df['title'].str.contains(i),"title_encode_Braces"] = 1
    pre_df.loc[pre_df['title_encode_Braces'].isnull(),"title_encode_Braces"] = 0
    
    #バズりやすい言葉があれば1,それ以外を0
    buzz_word = ["vs","www","巨大","大量"]
    for n in buzz_word:
        pre_df.loc[pre_df['title'].str.contains(n),"title_encode_word"] = 1
    pre_df.loc[pre_df['title_encode_word'].isnull(),"title_encode_word"] = 0
    
    #タイトルの長さ
    pre_df.loc[pre_df['len_title'] > 28,"title_encode_len"] = 2
    pre_df.loc[pre_df['len_title'] < 28 ,"title_encode_len"] = 1
    pre_df.loc[pre_df['len_title'] < 22,"title_encode_len"] = 0
    
    #100万再生なら1,違うなら0とする
    pre_df.loc[pre_df['viewCount'] > 3000000,"view_encode"] = 1
    pre_df.loc[pre_df['view_encode'].isnull(),"view_encode"]= 0
    
    #データフレームを整える
    fix_df = pre_df[['title_encode_Braces','title_encode_word','title_encode_len']]
    fix_df = fix_df.fillna(0)
    return fix_df

def decisiontree(df):
   
    X = preprocessingDTree(df)
    Y = df['view_encode']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    
    # 決定木インスタンス(木の深さ3)
    model = DecisionTreeClassifier(max_depth=8)
    #学習モデル構築。引数に訓練データの特徴量と、それに対応したラベル
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # .scoreで正解率を算出。
    print("train score:",model.score(X_train,y_train))
    print("test score:",model.score(X_test,y_test))
    plot_cm(predicted,y_test)
    
def plot_cm(predict,ytest):
    plot_cm = confusion_matrix(predict,ytest)
    sns.heatmap(plot_cm, annot=True, cmap='Reds')
    
decisiontree(df_Hikakin)
