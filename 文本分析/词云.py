import wordcloud
import jieba
import jieba.analyse
import pandas as pd

data=pd.read_excel("D:/京东商品信息.xls")

name=data.T.values[1]
tag=jieba.analyse.extract_tags(" ".join(name),10)

font = r'C:\Windows\Fonts\FZSTK.TTF'
w=wordcloud.WordCloud(font_path=font,background_color='white')
w.generate(" ".join(tag))
w.to_file("picture.png")