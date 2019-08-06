## 概要
YouTubeにある動画の中で[HikakinTV](https://www.youtube.com/user/HikakinTV)がどのような動画でバズるのか予測分析する。回帰分析や決定木を用いてバズるための特徴量を探し、その特徴量が多く含まれるジャンルの動画はどのようなものなのかをクラスタリングする。

## 使用する環境
1. Python 3.7
2. Jupyter Notebook

## 使用するライブラリ
今回のコードを動かすに当たってPythonのライブラリを用いる。
1. Pandas
2. matplotlib  
3. seaborn
4. sklearn
5. janome

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
2. LinearRegression.py  
3. Classification.py
4. morphogical.py
5. plotScatterMatrix.py
