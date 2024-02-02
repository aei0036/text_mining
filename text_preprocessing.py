#===================================================================
#  import
#===================================================================
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import Text
import numpy as np
import pandas as pd
import time
from gensim.models import Word2Vec
from gensim.models import CoherenceModel
from gensim import corpora
import gensim

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import os

os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
#from konlpy.tag import Okt
#from konlpy.tag import Kkma


#===================================================================
#  Global Value
#===================================================================

# 분석 대상 필드(분석 대상 엑셀 1행과 명칭 통일 / '국가코드'는 고정(영어/한국어 파악을 위한 필드))
target_field = ['국가코드','발명의 명칭','요약','대표청구항']

#엑셀 파일 경로
file_name='C:/Users/김수정/Documents/GitHub/text_mining/test_data_game.xlsx'

# 사용할 품사 선택
select_pos_tag = ['NN', 'JJ', 'NNS']

def preprocess(text):
  words = []
  
  try:
    words = list(text.keys())
  except AttributeError:
    two_dim_words = []
    for i in range(len(text)):
      two_dim_words.append(list(text[i].keys()))
    words = [item for sublist in two_dim_words for item in sublist]

  word_to_id = {}
  id_to_word = {}
  for word in words:
    if word not in word_to_id:
      new_id = len(word_to_id)
      word_to_id[word] = new_id
      id_to_word[new_id] = word

  #corpus = np.array([word_to_id[w] for w in words])
    
  #return corpus, word_to_id, id_to_word
  return word_to_id, id_to_word


def patent_data_open():
  return pd.read_excel(file_name, usecols=target_field)

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

  #text_temp = [ 0 for i in range (len(target_field))]

  # 분석 대상 데이터 소문자로 통일
  for i in range (1, len(target_field)):
    text[i] = text[target_field[i]].str.lower()

  #대표청구항 문장 토큰화(word_tokenize()) 및 품사 매칭(pos_tag())
  for i in range (1, len(target_field)):
     text[f'{target_field[i]}_명사추출'] = text[i].apply(lambda x: pos_tag(word_tokenize(x)))
  
  #선택한 품사만 추출
  NN_words = [[] for i in range (len(text.index))]
  total_NN_words = []

  for i in range (len(text.index)):
    for j in range (1, len(target_field)):
      for word, pos in text[f'{target_field[j]}_명사추출'].values[i]:
        for sel_tag in select_pos_tag:
          if sel_tag in pos:
            total_NN_words.append(word)
            NN_words[i].append(word)

  
  #porter algorithm 기반 stemming
  stemmer = PorterStemmer()
  stemmer_words = [[] for i in range (len(text.index))]   #엑셀 행별로 작업
  total_stemmer_words = []  #엑셀 모든 행 키워드에 대해 작업

  for word in total_NN_words:
    new_word = stemmer.stem(word)
    total_stemmer_words.append(new_word)

  for i in range (len(NN_words)):
    for word in NN_words[i]:
      new_word = stemmer.stem(word)
      stemmer_words[i].append(new_word)

  #단어 원형으로 변형(stations -> station)
  '''lemmatizer = WordNetLemmatizer()
  lemmatized_words = [[] for i in range (len(text.index))]   #엑셀 행별로 작업
  total_lemmatized_words = []   #엑셀 모든 행 키워드에 대해 작업
  for word in total_NN_words:
    new_word = lemmatizer.lemmatize(word,'n')
    total_lemmatized_words.append(new_word)

  for i in range (len(NN_words)):
    for word in NN_words[i]:
      new_word = lemmatizer.lemmatize(word,'n')
      lemmatized_words[i].append(new_word)

  #print(lemmatized_words)'''

  #불용어 제거
  stop_words = set(stopwords.words('english'))
  result_words = [[] for i in range (len(text.index))]   #엑셀 행별로 작업
  total_result_words = []  #엑셀 모든 행 키워드에 대해 작업
  for word in total_stemmer_words:
    if len(word) > 2:   #길이가 2 이하인 단어 제거
        if word not in stop_words:
            total_result_words.append(word)

  #길이가 2 이하인 단어 제거
  for i in range(len(stemmer_words)):
    for word in stemmer_words[i]:
      if len(word) > 2:
        if word not in stop_words:
          result_words[i].append(word)

  #단어 빈도수 체크
  total_vocab = {}
  sort_vocab = [[] for i in range (len(result_words))]
  vocab = [{} for i in range (len(result_words))]

  #모든 특허에 대하여 카운트
  for word in total_result_words:
    if word not in total_vocab:
      total_vocab[word] = 0
    total_vocab[word] += 1
    total_sort_vocab = dict(sorted(total_vocab.items(), key=lambda x: x[1], reverse=True))

  #각 특허별로 카운트
  for i in range(len(result_words)):
    for word in result_words[i]:
      if word not in vocab[i]:
        vocab[i][word] = 0
      vocab[i][word] += 1
      sort_vocab[i] = dict(sorted(vocab[i].items(), key=lambda x: x[1], reverse=True))
  
  #각 특허별로 빈도수가 2 이하인 단어 제거
  '''filtered_data = {}
  filtered_vocab = []
  for i in range(len(sort_vocab)):
    filtered_data = {key: value for key, value in sort_vocab[i].items() if value > 2}
    filtered_vocab.append(filtered_data)'''

  result_token = [total_sort_vocab, sort_vocab]

  return result_token

def tokenize_us(text):

  total_sentence = ['' for i in range (len(text.index))]

  for i in range (1, len(target_field)):
    text[i] = text[target_field[i]].str.lower()

  #문장 토큰화(word_tokenize()) 및 품사 매칭(pos_tag())
  text_to_sentence = []

  #문장 단위로 분할
  for i in range (1, len(target_field)):
    for j in range (len(text.index)):
      text_to_sentence.append(text[i].values[j].replace(':','.').replace(',','.').replace(';','.').split('.'))
  
  #문장을 단어로 분할
  sentence_to_token = ['' for i in range (len(text_to_sentence))]
  for i in range (len(text_to_sentence)):
    sentence_to_token[i] = str(text_to_sentence[i]).replace('[','').replace(']','').replace("'",'').replace(",",'').split(' ')
  
  #porter algorithm 기반 stemming
  stemmer = PorterStemmer()
  stemmer_words = [[] for i in range (len(sentence_to_token))]   #엑셀 행별로 작업
  for i in range(len(sentence_to_token)):
    for word in sentence_to_token[i]:
      new_word = stemmer.stem(word)
      stemmer_words[i].append(new_word)

  #불용어 제거
  stop_words = set(stopwords.words('english'))
  result_words = [[] for i in range (len(stemmer_words))]   #엑셀 행별로 작업

  #길이가 2 이하인 단어 및 불용어 제거
  for i in range(len(stemmer_words)):
    for word in stemmer_words[i]:
      if len(word) > 2:
        if word not in stop_words:
          result_words[i].append(word)

  #for i in range(len(result_words)):
  #  print(i, result_words[i])

  return result_words

def LDA_model(corpus, id_to_word, sentence_to_token_us):

  tokens = []

  for i in range(len(sentence_to_token_us)):
    tokens.append(list(sentence_to_token_us[i].keys()))
  #print(id_to_word.keys())
  # dictionary.doc2bow(token)을 통해 만드는 corpus 랑 내가 만드는 corpus가 불일치
    
  coherence_score = []
  #print(tokens)
  dictionary = corpora.Dictionary(tokens)

  #dictionary.filter_extremes(no_below=2) # 빈도 2미만 단어 제거
  print(len(dictionary.token2id))
  #corpus2 = [dictionary.doc2bow(token) for token in tokens] 
  #print(corpus)

  for i in range(2,10):

    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=10)
    coherence_model = CoherenceModel(model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    print('k=',i,'\nCoherence Score: ', coherence_lda)
    coherence_score.append(coherence_lda)

  #가장 높은 토픽수로 최종훈련
  max_topic = max(coherence_score)
  max_topic_index = coherence_score.index(max_topic)
  print("max_topic_index : ",max_topic_index)

  final_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=max_topic_index+2, passes=10)
  print(final_model.print_topics())
  
  prepared_data = gensimvis.prepare(topic_model=final_model, corpus=corpus, dictionary=dictionary)

  pyLDAvis.display(prepared_data)
  # 훈련된 모델에서 각 문서의 토픽 분포 얻기
  #topic_distribution = [final_model.get_document_topics(doc) for doc in corpus]

  # 각 문서를 가장 관련성 높은 토픽으로 분류
  #classified_topics = [max(dist, key=lambda x: x[1])[0] for dist in topic_distribution]

  # 분류 결과를 DataFrame에 추가
  #print("[classified_file] \n", classified_topics)

def vec_us(text):

  model = Word2Vec(text, vector_size=300, window=7, min_count=5, workers=1)
  word_vectors = model.wv
  #vocabs = word_vectors.vocab.keys()
  #word_vectors_list = [word_vectors[v] for v in vocabs]
  model_result = model.wv.most_similar("game")
  #print(model_result)

# token 변경 ('단어 : 빈도' 로 구성된 dict의 리스트)를 ((id, 빈도)로 구성된 튜플의 리스트)로 변경
def wti(word_to_id, token):
  
  new_token = []
  temp_tuple = ()
  for j in range (len(token)):
    temp_token = []
    for key in token[j].keys():
      temp_tuple = (word_to_id[key], token[j][key])
      temp_token.append(temp_tuple)
    new_token.append(temp_token)
  
  return new_token


def job():
  
  #실행시간체크시작
  start_time = time.time()
  
  # 특허 excel data 읽어옴
  patent_text = patent_data_open()
  
  # excel data 한국어/영어 구분
  patent_text_kr = language_type_filter(patent_text,'kr')
  patent_text_us = language_type_filter(patent_text,'us')

  # 한국어 명사 빈도 추출
  #token_kr(patent_text_kr)

  # 영어 토큰화 및 단어사용빈도 카운팅
  sentence_to_token_count_us = token_us(patent_text_us)

  #print(sentence_to_token_us[0]) #문서 전체 통합 토큰화
  #print(sentence_to_token_us[1]) #특허별로 토큰화

  # word : id dictionary 생성
  word_to_id, id_to_word = preprocess(sentence_to_token_count_us[0])

  # 단어 -> id로 변환
  corpus = wti(word_to_id, sentence_to_token_count_us[1])
  
  LDA_model(corpus, id_to_word, sentence_to_token_count_us[1])
  #활용 단어 사전 형성(단어 - id 매칭 / corpus : 단어id목록)
  #corpus, word_to_id, id_to_word = preprocess(sentence_to_token_us[0])

  # 영어 토큰화 및 word2vec 라이브러리 사용

  #tokenized_us = tokenize_us(patent_text_us)
  #vec_us(tokenized_us)


  #print(word_to_id)
  #print(id_to_word)

  #실행시간체크종료
  end_time = time.time()
  print("러닝타임 :",end_time - start_time)

if __name__ == '__main__':
  job()
