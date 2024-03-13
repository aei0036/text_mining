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
import re
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#===================================================================
#  Global Value
#===================================================================

# 분석 대상 필드(분석 대상 엑셀 1행과 명칭 통일 / '국가코드'는 고정(영어/한국어 파악을 위한 필드))
TARGET_FIELD = ['국가코드','발명의 명칭','요약','대표청구항']

#엑셀 파일 경로
IN_FILE_PATH='C:/Users/김수정/Documents/GitHub/text_mining/test_data_carbon_reduce.xlsx'
OUT_FILE_PATH = 'C:/Users/김수정/Documents/GitHub/text_mining/output_file.xlsx'
LDA_HTML_PATH = 'C:/Users/김수정/Documents/GitHub/text_mining/lda.html'


# 사용할 품사 선택..
SELECT_POS_TAG = ['NN', 'JJ', 'NNS', 'VB', 'VBD', 'VBG', 'VBN']

#최소 토픽수
MINI_TOPIC = 2

#코사인 유사도 결과값 기반 노이즈 판단 기준
#유사도 최소값
NOISE_SIM_MIN_VALUE1 = 0.5
#유사도 최소값(해당 토픽에 해당하는 유사도 최소값)
NOISE_SIM_MIN_VALUE2 = 0.5
#유사도 최소값을 만족하는 최소 토픽 수
NOISE_TOPIC_MIN_VALUE = 2

#특허 불용어
PATENT_STOP_WORDS = ['said','provid','compris','least','includ','wherein','configur','method','process','use','determin','system','devic','unit','element']



#===================================================================
#  Function
#===================================================================

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

def language_type_filter(patent_text, lan_type):
  if (lan_type=='kr'):
    patent_text_filter = patent_text.loc[(patent_text['국가코드'] == 'KR') | (patent_text['국가코드'] == 'JP')]
  elif(lan_type=='us'):
    patent_text_filter = patent_text.loc[(patent_text['국가코드'] == 'US') | (patent_text['국가코드'] == 'EP') | (patent_text['국가코드'] == 'CN')]
  else:
    print('language_type_error!!!')
    patent_text_filter = ''

  return patent_text_filter


def token_us(text):

  text_temp = pd.DataFrame()
  
  # 분석 대상 데이터 소문자로 통일   b
  for i in range (1, len(TARGET_FIELD)):
    text_temp[i] = text[TARGET_FIELD[i]].str.lower()
  
  #대표청구항 문장 토큰화(word_tokenize()) 및 품사 매칭(pos_tag())
  for i in range (1, len(TARGET_FIELD)):
     text_temp[f'{TARGET_FIELD[i]}_품사추출'] = text_temp[i].apply(lambda x: pos_tag(word_tokenize(x)))
  

  #선택한 품사만 추출
  NN_words = [[] for i in range (len(text.index))]
  total_NN_words = []

  for i in range (len(text_temp.index)):
    for j in range (1, len(TARGET_FIELD)):
      for word, pos in text_temp[f'{TARGET_FIELD[j]}_품사추출'].values[i]:
        for sel_tag in SELECT_POS_TAG:
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
  print("불용어 제거")
  stop_words = set(stopwords.words('english'))
  stop_words.update(PATENT_STOP_WORDS)

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
  print("빈도수 체크 및 소팅")
  for word in total_result_words:
    if word not in total_vocab:
      total_vocab[word] = 0
    total_vocab[word] += 1
    #total_sort_vocab = dict(sorted(total_vocab.items(), key=lambda x: x[1], reverse=True))
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

  print(total_sort_vocab)
  return result_token

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


def LDA_model(sentence_to_token_us, min_topic):

  tokens = []

  for i in range(len(sentence_to_token_us)):
    tokens.append(list(sentence_to_token_us[i].keys()))
    
  coherence_score = []
  dictionary = corpora.Dictionary(tokens)

  #dictionary.filter_extremes(no_below=2) # 빈도 2미만 단어 제거
  print("words num : ", len(dictionary.token2id))

  #아래 함수 사용하면 빈도수가 모두 1인 corpus가 생성됨.... 그래서 아래 wti함수 활용
  #corpus2 = [dictionary.doc2bow(token) for token in tokens] 
  corpus = wti(dictionary.token2id, sentence_to_token_us)

  for i in range(min_topic,min_topic + 10):
  
    model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, passes=15)
    coherence_model = CoherenceModel(model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    print('k=',i,'\nCoherence Score: ', coherence_lda)
    coherence_score.append(coherence_lda)

  if (max(coherence_score)<0.55):
    print("응집도 낮음. 불용어 추가 필요")

  #가장 높은 토픽수로 최종훈련
  max_topic = max(coherence_score)
  max_topic_index = coherence_score.index(max_topic)

  final_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=max_topic_index + min_topic, passes=10)
  
  prepared_data = gensimvis.prepare(topic_model=final_model, corpus=corpus, dictionary=dictionary)
  #pyLDAvis.display(prepared_data)
  #pyLDAvis.enable_notebook(local=True)
  pyLDAvis.save_html(prepared_data,LDA_HTML_PATH)
  #pyLDAvis.show(prepared_data)

  # 훈련된 모델에서 각 문서의 토픽 분포 얻기
  topic_distribution = [final_model.get_document_topics(doc) for doc in corpus]

  # 각 문서를 가장 관련성 높은 토픽으로 분류
  classified_topics = [max(dist, key=lambda x: x[1])[0] for dist in topic_distribution]

  # 분류 결과를 DataFrame에 추가
  #print("[classified_file] \n", classified_topics)
  return final_model.print_topics(num_words=20), topic_distribution, classified_topics

def vec_us(text):

  model = Word2Vec(text, vector_size=300, window=7, min_count=5, workers=1)
  word_vectors = model.wv
  #vocabs = word_vectors.vocab.keys()
  #word_vectors_list = [word_vectors[v] for v in vocabs]
  model_result = model.wv.most_similar("game")
  #print(model_result)

def patent_data_out(patent, topic_word, classified_topics, noise_check, cos_sim, topic_distribution):
  
  #분석 데이터를 기존 df(특허 엑셀 데이터)의 맨 앞에 붙임
  df_patent = pd.DataFrame(patent)
  df_patent.insert(0, 'topic', classified_topics)
  df_patent.insert(1, 'topic_dist', topic_distribution)
  df_patent.insert(2, 'NOISE', noise_check)
  for i in range(len(cos_sim)):
    df_patent.insert(i+3, 'cos_similary_'+str(i), cos_sim[i])
  
  # 토픽 결과를 데이터프레임으로 변환
  columns = ['Topic', 'Keywords']
  df_topics = pd.DataFrame(topic_word, columns=columns)

  with pd.ExcelWriter(OUT_FILE_PATH, engine='xlsxwriter', mode='w') as writer:
    # patent 데이터프레임을 'DATA' 시트에 작성
    df_patent.to_excel(writer, sheet_name='DATA', index=False)

    # topic정보에 대한 데이터프레임을 'TOPIC' 시트에 작성
    df_topics.to_excel(writer, sheet_name='TOPIC', index=False)


# keyword기반 코사인 유사도 분석
def keywords_cosine_similary(sentence_to_token_count_us, topic_word):

  #문헌별로 키워드를 추출하여 문자열로 변환한후 keyword_list에 저장
  keyword_list = []
  topic_list = []
  for i in range(len(sentence_to_token_count_us)):
    #키워드를 list로 추출
    temp_list = list(sentence_to_token_count_us[i].keys())

    #하나의 문자열로 합치기
    keyword_list_str = ' '.join(temp_list)

    keyword_list.append(keyword_list_str)

  #토픽별로 키워드를 추출하여 topic_words_list에 저장
  for i in range(len(topic_word)):
    #키워드를 list로 추출( " " 사이 문자를 리스트로 뽑아내는 정규표현식)
    temp_list = re.findall(r'"(.*?)"', topic_word[i][1])
    
    #하나의 문자열로 합치기
    topic_words_list_str = ' '.join(temp_list)
    
    
    topic_list.append(topic_words_list_str)

  cosine_similarities = []
  for i in range(len(topic_list)):
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix_topic = vectorizer.fit_transform([topic_list[i]])
    tfidf_matrix_keyword = vectorizer.transform(keyword_list)
    
    # 코사인 유사도 계산
    cosine_similarities.append(cosine_similarity(tfidf_matrix_topic, tfidf_matrix_keyword))

    # 어레이를 2차원 리스트로 변환
    cosime_sim_list = [arr[0].tolist() for arr in cosine_similarities]
  return cosime_sim_list
    # 결과 출력
    #for i, similarity in enumerate(cosine_similarities[0]):
    #    print(keyword_list[i])
    #    print(f"코사인 유사도 (topic_word와 test_word[{i}]): {similarity}")


#유사도값 기반 노이즈 판단
def noise_check_func(cos_sim, classified_topics):
  noise_check = []

  #특허에 해당하는 토픽과의 유사도값이 0.5이상인 경우 TRUE, 미만이면 NOISE
  for i in range(len(cos_sim[0])):
    if (cos_sim[classified_topics[i]][i] > NOISE_SIM_MIN_VALUE2):
      noise_check.append("TRUE")
    else: noise_check.append("NOISE")

  '''for col in zip(*cos_sim):
    # 현재 열에서 0.5 이상인 값의 개수 세기
    count_above_0_5 = sum(value >= NOISE_SIM_MIN_VALUE1 for value in col)
        
    # 0.5 이상인 값이 2개 이상인 경우 True, 그렇지 않으면 False 저장
    if (count_above_0_5 >= NOISE_TOPIC_MIN_VALUE):
      noise_checkc
    else: noise_check.append("NOISE")

  for i in range(len(noise_check)):
    if (noise_check[i]=="TRUE") and (cos_sim[classified_topics[i]][i] > NOISE_SIM_MIN_VALUE2):
      noise_check[i]="TRUE"
    else: noise_check[i]="NOISE"'''
  

  return noise_check

def job():
  
  #실행시간체크시작
  start_time = time.time()
  
  # 특허 excel data 읽어옴
  patent_text = pd.read_excel(IN_FILE_PATH)

  # excel data 한국어/영어 구분
  patent_text_kr = language_type_filter(patent_text,'kr')
  patent_text_us = language_type_filter(patent_text,'us')

  # 한국어 명사 빈도 추출
  #token_kr(patent_text_kr)

  # 영어 토큰화 및 단어사용빈도 카운팅
  print("문장 토큰화 수행")
  sentence_to_token_count_us = token_us(patent_text_us)
  print("문장 토큰화 종료")
  # word : id dictionary 생성
  #word_to_id, id_to_word = preprocess(sentence_to_token_count_us[0])
  min_topic = MINI_TOPIC
  topic_word, topic_distribution, classified_topics = LDA_model(sentence_to_token_count_us[1], min_topic)

  cos_sim = keywords_cosine_similary(sentence_to_token_count_us[1], topic_word)

  noise_check = noise_check_func(cos_sim, classified_topics)
  # excel 출력
  patent_data_out(patent_text_us, topic_word, classified_topics, noise_check, cos_sim, topic_distribution)
  
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
