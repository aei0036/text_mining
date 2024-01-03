from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import Text
import pandas as pd
#from konlpy.tag import Okt
#from konlpy.tag import Kkma


def patent_data_open():
  file_name='C:/Users/김수정/Documents/GitHub/text_mining/test_data.xlsx'
  return pd.read_excel(file_name, usecols="A,H")

def language_type_filter(patent_text, lan_type):
  if (lan_type=='kr'):
    patent_text_filter = patent_text.loc[(patent_text['국가코드'] == 'KR') | (patent_text['국가코드'] == 'JP')]
  elif(lan_type=='us'):
    patent_text_filter = patent_text.loc[(patent_text['국가코드'] == 'US') | (patent_text['국가코드'] == 'EP') | (patent_text['국가코드'] == 'CN')]
  else:
    print('language_type_error!!!')
    patent_text_filter = ''

  return patent_text_filter


#def token_kr(text):
#  okt = Okt()
#  kkma = Kkma()

def token_us(text):

  #소문자로 통일
  text['대표청구항'] = text['대표청구항'].str.lower()

  #대표청구항 문장 토큰화(word_tokenize()) 및 품사 매칭(pos_tag())
  text['명사추출'] = text['대표청구항'].apply(lambda x: pos_tag(word_tokenize(x)))

  #명사만 추출
  NN_words = []
  for i in range (len(text.index)):
    for word, pos in text['명사추출'].values[i]:
      if 'NN' in pos:
        NN_words.append(word)

  #단어 원형으로 변형(stations -> station)
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = []
  for word in NN_words:
    new_word = lemmatizer.lemmatize(word,'n')
    lemmatized_words.append(new_word)


  #불용어 제거
  stop_words = set(stopwords.words('english'))
  result_words = []
  for word in lemmatized_words:
    if len(word) > 2:   #길이가 2 이하인 단어 제거
        if word not in stop_words:
            result_words.append(word)

  #단어 빈도수 체크
  vocab = {}
  for word in result_words:
    if word not in vocab:
      vocab[word] = 0
    vocab[word] += 1

    sort_vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))


  print(sort_vocab)
  print(len(text.index))


def job():
  # 특허 excel data 읽어옴
  patent_text = patent_data_open()

  # excel data 한국어/영어 구분
  patent_text_kr = language_type_filter(patent_text,'kr')
  patent_text_us = language_type_filter(patent_text,'us')

  # 한국어 명사 빈도 추출
  #token_kr(patent_text_kr)

  # 영어 명사 빈도 추출
  token_us(patent_text_us)

if __name__ == '__main__':
  job()
