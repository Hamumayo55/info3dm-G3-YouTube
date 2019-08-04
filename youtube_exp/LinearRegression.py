from quilt.data.haradai1262 import YouTuber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)

def plot_ScatterMatrix(df):
    
    #タイトルの長さを確認
    df['len_title'] = df['title'].apply(lambda x: len(str(x).replace(' ', '')))
    
    #散布図行列と相関行列の作成
    cols = ['viewCount','likeCount','dislikeCount','len_title']
    
    sns.pairplot(df[cols], size=2.5)
    plt.tight_layout()
    # plt.savefig('images/10_03.png', dpi=300)
    plt.show()
    
    cm = np.corrcoef(df[cols].values.T)
    #sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 15},
                     yticklabels=cols,
                     xticklabels=cols)
    
    plt.tight_layout()
    # plt.savefig('images/10_04.png', dpi=300)
    plt.show()

def LinearRegression(df):
    
    #データの準備
    df_explain = df[['likeCount','dislikeCount','len_title']]
    df_target = df[['viewCount']]
    
    clf = linear_model.LinearRegression() 
    
    clf.fit(df_explain,df_target)
    print('intercept b = ', clf.intercept_)
    
    y_pred = clf.predict(df_explain)
    mse = mean_squared_error(df_target, y_pred)
    print('MSE = ', mse)
    print('R^2 = ', clf.score(df_explain,df_target))
    
    X_train, X_test, y_train, y_test = train_test_split(df_explain,df_target, random_state=0)
    print('train = ', len(X_train))
    print('test =', len(X_test))
    
    clf.fit(X_train, y_train)
    print('Training Score',clf.score(X_train, y_train))
    print('Test Score',clf.score(X_test, y_test))
    
plot_ScatterMatrix(df_Hikakin)
LinearRegression(df_Hikakin)