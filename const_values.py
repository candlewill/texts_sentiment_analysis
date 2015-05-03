__author__ = 'NLP-PC'
# 表情符号替换，只采用那些明显带有正面或者负面情感的词语来替代
emo_repl = {
    #正面情感的表情
    '&lt;3': 'good',
    ':d': 'good',
    ':dd': 'good',
    '8)': 'good',
    ':-)': 'good',
    ':)': 'good',
    ';)': 'good',
    '(-:': 'good',
    '(:': 'good',

    #负面情感的表情
    ':/': 'bad',
    ':&gt;': 'sad',
    ":')": 'sad',
    ":-(": 'bad',
    ':(': 'bad',
    ':S': 'bad',
    ':-S': 'bad',
}
#确保:DD在:D之前替换，即先替换:DD，后替换:D
emo_repl_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in emo_repl.keys()]))]

#利用正则表达式及其扩展（\b标记出词语边界）来定义那些缩写形式
re_repl = {
    #缩写替换
    r"\br\b": "are", # r -> are
    r"\bu\b": "you", # u -> you
    r"\bhaha\b": "ha", # haha -> ha
    r"\bhahaha\b": "ha", # hahaha -> ha
    r"\bdon't\b": "do not", # don't -> do not
    r"\bdoesn't\b": "does not", # dosen't -> does not
    r"\bdidn't\b": "did not", # didn't -> did not
    r"\bhasn't\b": "has not", # hasn't -> has not
    r"\bhaven't\b": "have not", # haven't -> have not
    r"\bhadn't\b": "had not", # hadn't -> had not
    r"\bwon't\b": "will not", # won't -> will not
    r"\bwouldn't\b": "would not", # woudn't  -> would not
    r"\bcan't\b": "can not", # can't -> can not
    r"\bcannot\b": "can not", # cannot -> can not

    #利用正则表达式，替换其他
    r'((www\.[^\s]+)|(https?://[^\s]+))':'url', #www.baidu.com or http: -> url
    r'@[^\s]+': 'someone', # @someone -> someone
    r'#([^\s]+)':r'\1', # #topic -> topic
    r"(.)\1{1,}":r"\1\1", # noooooope -> noope
    r" +": " ", # 删去多余空格
    r"\bhttp[^\s]+\b":'url', #以http开始的单词 ->url
}