from quilt.data.haradai1262 import YouTuber
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
用意したデータフレームを引数としてplot_ScatterMatrixに
渡すと散布図行列と相関行列を作成するプログラム
"""

#データフレームの準備
df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)

def plot_ScatterMatrix(df):
    """
    散布図行列と相関行列を作成し、プロットする関数
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    """
    
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
    
plot_ScatterMatrix(df_Hikakin)
