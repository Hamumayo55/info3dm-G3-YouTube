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
from janome.tokenizer import Tokenizer
import collections

"""
データの読み込み処理
チャンネルID(cid)でヒカキンの動画のみとしている
"""
df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)

def morphological(df_fram):

    """
    タイトルの単語を分けてワードカウントを行う
    ----------
    Parameters
    df_fram : pandas.core.frame.DataFrame
    ----------
    Return
    c：カウントしたワードの辞書
    ----------
    """

    df_OneMillion = df_fram[df_fram["viewCount"] >3000000]
     
    list_tword = []
    t = Tokenizer()
    for i in range(len(df_OneMillion)):
        title = df_OneMillion.iloc[i,1]
        for token in t.tokenize(title):
            list_tword.append((token.surface))
            c = collections.Counter(list_tword)
            
    return print(c)

def preprocessingDTree(pre_df):

    """
    決定木を作成するために必要な前処理を行う関数
        - 行う前処理
            1. タイトル内に'[]'が含まれているかを1,0で判定
            2. タイトル内に特定のワードが含まれているかを1,0で判定
            3. タイトルの長さを3つに分ける
            4. 300万再生以上なら1,その他は0とする
    ----------
    Parameters
    df：pandas.core.frame.DataFrame
    ----------
    Return 
    fix_df：前処理を行なったデータフレーム
    """
    
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
    
    #300万再生なら1,違うなら0とする
    pre_df.loc[pre_df['viewCount'] > 3000000,"view_encode"] = 1
    pre_df.loc[pre_df['view_encode'].isnull(),"view_encode"]= 0
    
    #データフレームを整える
    fix_df = pre_df[['title_encode_Braces','title_encode_word','title_encode_len']]
    fix_df = fix_df.fillna(0)
    return fix_df

def decisiontree(df):

    """
    決定木を実装している
    preprocessingDTree()で処理されたデータ
        - 特徴量(title_encode_Braces','title_encode_word','title_encode_len)
        - 教師データ(再生数300万以上なら1,それ以下なら0)
    ----------
    Parameters
    df：pandas.core.frame.DataFrame
    """
   
    X = preprocessingDTree(df)
    Y = df['view_encode']
    #データの分割処理
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    
    # 決定木インスタンス(木の深さ8)
    model = DecisionTreeClassifier(max_depth=8)
    #学習モデル構築。引数に訓練データの特徴量と、それに対応したラベル
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # .scoreで正解率を算出。
    print("train score:",model.score(X_train,y_train))
    print("test score:",model.score(X_test,y_test))
    plot_cm(predicted,y_test)
    
def plot_cm(predict,ytest):

    """
    混同行列を作成する
    ----------
    Parameters
    predict：numpy.ndarray
    ytest：pandas.core.series.Series
    """
    plot_cm = confusion_matrix(predict,ytest)
    sns.heatmap(plot_cm, annot=True, cmap='Reds')
    
morphological(df_Hikakin)
decisiontree(df_Hikakin)
