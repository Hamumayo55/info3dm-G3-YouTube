from quilt.data.haradai1262 import YouTuber

# HIKAKINの動画のみを抽出します。
df_bigi = YouTuber.channel_videos.UUUM_videos()
df_bigi = df_bigi[df_bigi['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)
df = df_bigi.loc[:,["title","tags"]]


from janome.tokenizer import Tokenizer
tokenizer = Tokenizer()




def get_noun( str ):
    """
    タイトルから名詞を抽出する
    """
    nouns = []
    for token in tokenizer.tokenize( str ):
        pos = token.part_of_speech.split(',')[0]
        if pos == '名詞': nouns.append( token.base_form )
    return nouns

def tag2list( str ):
    """
    タグを結合する
    """
    tags = []
    for i in str[1:-1].split(','):
        i = i.replace(' ','')
        tags.append( i[1:-1] )
    return tags

df['title_noun'] = df['title'].apply( get_noun )
df['tags_noun'] = df['tags'].apply( tag2list )
X_tmp = df.loc[:,['title_noun','tags_noun']].values
X = [ i[0] + i[1] for i in X_tmp ]

#作成したコーパスにLSIを適用し、各映像の特徴量を抽出します。
#LSIには、gensimを使用しており、"dictionary.filter_extremes(no_below=2, no_above=0.5 )"で語彙を調整しています。

import gensim
from gensim import corpora, models, similarities
import numpy as np

dictionary = corpora.Dictionary( X )
dictionary.filter_extremes(no_below=2, no_above=0.5 )
corpus = [dictionary.doc2bow(text) for text in X]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=30)
lsi_feature = lsi[corpus]

X_lsi = np.ndarray(( len(corpus), 30 ))
for i in range( len( corpus ) ):
    for j in lsi_feature[i]:
        X_lsi[i, j[0]] = j[1]

def get_groups( t_cluster ):
    """
    得られた特徴凌を用いてクラスタリングを行う
    K-means法
    クラスタ数 15
    """
    groups_tmp = {}
    for i, l in enumerate( t_cluster ):
        if not l in groups_tmp: groups_tmp[l] = []
        groups_tmp[l].append( i )
    return groups_tmp

from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=15, random_state=0).fit(X_lsi)
group = get_groups( cluster.labels_ )

import pandas as pd
def get_cluster_means( x_tmp, group_tmp ):
    """
    クラスタを可視化する
    """
    mean_df_tmp= x_tmp.mean()
    mean_df_tmp.name = -1
    for i in range( len( group_tmp ) ):
        c = pd.DataFrame(x_tmp,index=group_tmp[i])
        c_mean = c.mean()
        c_mean.name = i
        mean_df_tmp = pd.concat([mean_df_tmp, c_mean], axis=1)
    mean_df_tmp = mean_df_tmp.drop( columns=[-1] )
    return mean_df_tmp

df_stat = df_bigi.loc[:,['viewCount','dislikeCount']]
mean_df = get_cluster_means( df_stat, group ).T
mean_df.style.background_gradient( cmap='Oranges' )

print("各クラスタの統計値の平均値")
print(mean_df)
