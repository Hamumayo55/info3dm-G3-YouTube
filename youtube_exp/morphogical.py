from quilt.data.haradai1262 import YouTuber
from janome.tokenizer import Tokenizer
import collections

"""
形態素解析でタイトルによく出てくるワードを
カウントするプログラム
"""

df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)

def morphological(df_fram):

    """
    タイトルの単語を分けてワードカウントを行う

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    """
    
    df_fram['len_title'] = df_fram['title'].apply(lambda x: len(str(x).replace(' ', '')))
    df_OneMillion = df_fram[df_fram["viewCount"] >3000000]
     
    list_tword = []
    t = Tokenizer()
    for i in range(len(df_OneMillion)):
        title = df_OneMillion.iloc[i,1]
        for token in t.tokenize(title):
            list_tword.append((token.surface))
            c = collections.Counter(list_tword)
    print(c)

morphological(df_Hikakin)