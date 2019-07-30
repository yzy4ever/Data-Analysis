from wordcloud import WordCloud
import matplotlib.pyplot as plt

#plt.figure(figsize = (12, 8))

list_of_genres = [['Action', 'Adventure', 'Thriller', 'War'],
 ['Comedy', 'Crime'],
 ['Action'],
 ['Adventure', 'Family'],
 ['Romance', 'Comedy'],
 ['Romance', 'Comedy'],
 ['Crime', 'Drama', 'Mystery', 'Thriller'],
 ['Drama'],
 ['Documentary']]
text = ' '.join([i for j in list_of_genres for i in j])   #将list转换为一连串的字符串

wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Word')
plt.axis("off")
plt.show()