from quilt.data.haradai1262 import YouTuber
from janome.tokenizer import Tokenizer
import collections

"""
形態素解析でタイトルによく出てくるワードをカウントするプログラム
morphological(df_fram)では正しく動いているかをテストしている。
実装として使うときはdecision.pyのmorphological(df_fram)を参照
"""

df = YouTuber.channel_videos.UUUM_videos()
df_Hikakin = df[df['cid'] == 'UCZf__ehlCEBPop___sldpBUQ'].reset_index(drop=True)


def morphological(df_fram):

    """
    タイトルの単語を分けてワードカウントを行う
    ----------
    Args
    df_fram : pandas.core.frame.DataFrame
    ----------
    Return
    type(c)：True
    ----------
    >>> morphological(df_Hikakin)
    True
    """

    df_OneMillion = df_fram[df_fram["viewCount"] >3000000]
     
    list_tword = []
    t = Tokenizer()
    for i in range(len(df_OneMillion)):
        title = df_OneMillion.iloc[i,1]
        for token in t.tokenize(title):
            list_tword.append((token.surface))
            c = collections.Counter(list_tword)

    return issubclass(type(c), dict)

morphological(df_Hikakin)

if __name__ == "__main__":
    import doctest
    doctest.testmod()