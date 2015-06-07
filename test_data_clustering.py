__author__ = 'NLP-PC'

def expand_text_list(text_list, expand_content):
    out=[]
    for text in text_list:
        out.append(text + ' ' + expand_content[0])
    return out
# test
# text_list, expand_content=['we are ok', 'are you ok'], 'who are you and why are you here'
# print(expand_text_list(text_list, expand_content))