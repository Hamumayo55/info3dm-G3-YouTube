# Info3dm Group３ 成果物

## 概要
YouTubeにある動画の中で[HikakinTV](https://www.youtube.com/user/HikakinTV)がどのような動画でバズるのか予測分析する。回帰分析や決定木を用いてバズるための特徴量を探し、その特徴量が多く含まれるジャンルの動画はどのようなものなのかをクラスタリングする。

## 開発環境
1. Python 3.7
2. Anaconda
3. Jupyter Notebook

## 使用するライブラリ
今回のコードを動かすに当たってPythonのライブラリを用いる。
1. Pandas
2. matplotlib  
3. seaborn
4. sklearn
5. janome

## データセット
今回のデータセットは、[YouTuberデータセット公開してみた](https://qiita.com/myaun/items/7e0dd7f3f9d9d2fef497)を用いた。基本的なインストール方法は記載されている。

## 実行方法
1. *Jupyter Notebook*のファイルを実行する方法ためのコマンド(実行結果だけ見たいなら実行する必要はない)  
`$ pyenv local anaconda3-5.3.0`  
`$ nohup jupyter notebook &`  

2. *ファイル*(.py)実行方法  
youtube_expのフォルダ内まで移動してコマンド
`$ python 実行したいファイル`を行うと実行できる。

3. doctestの実行方法  
doctestの実装されているコード(morphogical.py)を実行するときは`$ python morphogical.py -v`で行う。

## 使用するコード
1. decision_tree.py 
    1. 決定木を用いて動画のタイトルと再生回数(バズり度)の関係を調べるためのコード  
2. LinearRegression.py  
    1. 目的変数と説明変数を用いて線形回帰を行う
3. Classification.py  
    1. 動画のタイトルとタグを元にクラスタリングを行う
4. morphogical.py  
    1. 動画のタイトルからよく使われれいる単語を形態素解析を行う
5. plotScatterMatrix.py  
    1. 散布図行列と相関行列を作成する
    
## 開発者および連絡先
氏名；新城 巧也  
連絡先：e175769@ie.u-ryukyu.ac.jp  

氏名：大城 龍太郎  
連絡先：e175739@ie.u-ryukyu.ac.jp


