from __future__ import print_function
import json
import os
import six
import paddlehub as hub

if __name__ == "__main__":
    # 加载senta模型
    senta = hub.Module(name="senta_bilstm")

    # 把要测试的短文本以str格式放到这个列表里
    test_text = [
        '小明虽然考了第一，但是他一点也不骄傲',  # 积极
        '你不是不聪明，而是不认真',  # 消极
        '虽然小明很努力，但是他还是没有考100分',  # 消极
        '虽然小明有时很顽皮，但是他很懂事',  # 积极
        '虽然这座桥已经建了很多年，但是她依然很坚固',  # 积极
        '他虽然很顽皮，但是学习很好',  # 积极
        '学习不是枯燥无味，而是趣味横生',  # 积极
        '虽然很困难，但是我还是不会退缩',  # 积极
        '虽然小妹妹只有5岁，但是她能把乘法口诀倒背如流',  # 积极
        '虽然我很过分，但是都是为了你好',  # 积极
        '小明成绩不好，不是因为不聪明，而是因为不努力',  # 消极
        '虽然这样做不妥当，但已经是最好的选择',  # 积极
        '这次虽然失败，但却是成功的开始',  # 积极
        '虽然这道题很难，但是我相信我会把它做出来',  # 积极
        '虽然爷爷已经很老了，但是他还是坚持每天做运动',  # 积极
        '不是没有美，而是我们缺少发现美的眼光',  # 消极
        '虽然他们有良好的生活条件，但是浪费资源迟早会带来恶果',  # 消极
        '他不是我们的敌人，而是我们的朋友',  # 积极
        '他不是不会做，而是不想做',  # 消极
        '虽然那个梦想看起来离我遥不可及，但是我相信经过我的努力它一定会实现',  # 积极
    ]

    # 指定模型输入
    input_dict = {"text": test_text}

    # 把数据喂给senta模型的文本分类函数
    results = senta.sentiment_classify(data=input_dict)

    # 遍历分析每个短文本
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print('text: {},\t  predict: {}'.format(results[index]['text'],results[index]['sentiment_key']))

