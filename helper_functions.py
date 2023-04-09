import pandas as pd
import numpy as np
# expand contractions
import contractions
# match regular expression
import re
# for tokenization
from nltk.tokenize import TweetTokenizer
# for POS tagging
from nltk.tokenize import sent_tokenize
from nltk import pos_tag_sents
# for lemmatization
from nltk.stem import WordNetLemmatizer
# for stop words
from nltk.corpus import stopwords
# for punctuations
import string
# for word frequency
from sklearn.feature_extraction.text import CountVectorizer
# for readability
from textatistic import Textatistic
# for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# for word embeddings
import gensim
# for evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
# for KF validation
from sklearn.model_selection import KFold


'''
data cleaning
'''
def data_cleaning(data):
    data2 = data.copy()
    
    # generate noise-free data
    print('removing noise')
    # remove non-ASCI characters
    data2["noise_removed_text"] = data2["comment_text"].apply(lambda x: re.sub(r'[^\x00-\x7f]', ' ', x))
    # expand contractions
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: contractions.fix(x))
    # remove URL
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', ' ', x))
    # remove special characters
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        u"\ufe0f"  # dingbats
        "]+", flags = re.UNICODE)
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: regrex_pattern.sub(' ', x))
    # remove extra spaces
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: re.sub('\r\n', ' ', x))
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: re.sub('\n', ' ', x))
    data2["noise_removed_text"] = data2["noise_removed_text"].apply(lambda x: re.sub(' +', ' ', x))
    
    # further cleaning
    print('further cleaning the text')
    # convert to lower case
    data2["clean_text"] = data2["noise_removed_text"].apply(lambda x: x.lower())
    # remove duplicated punctuations
    regrex_pattern2 = re.compile(r'''([!#$%&'()*+,./:;<=>?@[\]^_`{|}~-])[!"#$%&'()*+,./:;<=>?@[\]^_`{|}~-]+''')
    data2["clean_text"] = data2["clean_text"].apply(lambda x: regrex_pattern2.sub(r'\1', x))
    
    return data2


'''
data preprocessing
'''
# helper function to tokenize words in each sentence
def to_word_tokens(sent_tokens):
    tt = TweetTokenizer()
    word_tokens = [] 
    for sent_token in sent_tokens:
        word_tokens.append(tt.tokenize(sent_token))
    return word_tokens

def data_preprocessing(data):
    data2 = data.copy()
    
    # tokenization
    print('tokenizing the text')
    tt = TweetTokenizer()
    # with punctuations
    data2['text_tokenized'] = data2['clean_text'].apply(tt.tokenize)
    # without punctuations
    clean_text_nopunc = data2["clean_text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data2['text_tokenized_nopunc'] = clean_text_nopunc.apply(tt.tokenize)
    
    # POS tagging -- time consuming
    print('tagging the text')
    # tokenize sentences
    for_tagging_use_sent_token = data2['clean_text'].apply(sent_tokenize)
    # tokenize words in sentences 
    for_tagging_use_word_token = for_tagging_use_sent_token.apply(to_word_tokens)
    # perform POS tagging
    data2['POS_tagging'] = for_tagging_use_word_token.apply(pos_tag_sents)
    # flatten the POS tagging
    data2['POS_tagging_flat'] = data2['POS_tagging'].apply(lambda x: [element for innerList in x for element in innerList])
    
    # lemmatization
    print('lemmatizing the text')
    lemmatizer = WordNetLemmatizer()
    data2['lemmatization'] = data2['POS_tagging_flat'].apply(lambda x: [lemmatizer.lemmatize(word) for word, tag in x 
          if ((tag.startswith('JJ') or tag.startswith('NN') or tag.startswith('RB') or tag.startswith('VB')) and (word not in string.punctuation))])
    
    return data2


'''
feature engineering
'''
# helper function to get sentence length in words
def get_sentence_stats(text_tag):
    length = []
    for sent in text_tag:
        word_list=[]
        for word in sent:
            if word[0] not in string.punctuation:
                word_list.append(word[0])
        length.append(len(word_list))
    # dealing with zero length case
    if len(length)==0:
        return [0, 0, 0, 0]
    else:
        return [np.mean(length), np.max(length), np.min(length), np.std(length)]
    
# helper function to get punctuations
def get_punctuations(sentences):
    count = 0
    for punc in string.punctuation:
        count += sentences.count(punc) 
    return count

# helper function to get unique words and repeated words
def get_unique_repeated(sentences):
    try:
        vec = CountVectorizer(ngram_range=(1, 1), stop_words=None)
        bow = vec.fit_transform([sentences])
        sum_of_words = bow.sum(axis=0)
        unigrams_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        unigrams_freq = pd.DataFrame(unigrams_freq,columns=['word','freq'])
        num_unqiue = sum(unigrams_freq[unigrams_freq.freq==1].freq)
        num_repeated = len(unigrams_freq[unigrams_freq.freq>1])
        return [num_unqiue, num_repeated]
    except:
        # dealing with the case when there is only stop words in the text
        try:
            sentences = sentences.replace(' ', '')
            vec = CountVectorizer(ngram_range=(1, 1), stop_words=None)
            bow = vec.fit_transform([sentences])
            sum_of_words = bow.sum(axis=0)
            unigrams_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            unigrams_freq = pd.DataFrame(unigrams_freq,columns=['word','freq'])
            num_unqiue = sum(unigrams_freq[unigrams_freq.freq==1].freq)
            num_repeated = len(unigrams_freq[unigrams_freq.freq>1])
            return [num_unqiue, num_repeated]
        # dealing with empty text
        except:
            return [0, 0]
        
# helper function to get word tag counts
def get_word_tag_count(text_tag):
    noun_count = len([w for w in text_tag if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in text_tag if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in text_tag if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return [noun_count, adjective_count, verb_count]

# helper function to get the uppercase features
def get_uppercase(text):
    uppercase_count = sum(1 for c in text if c.isupper())
    tt = TweetTokenizer()
    text_tokenized = tt.tokenize(text)
    uppercase_words = sum(1 for word in text_tokenized if word.isupper())
    return [uppercase_count, uppercase_words]

# helper function to get the number of occurence of a regex
def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))

# helper function to get the readability score
def readability(text):
    # hyphenated word cannot have more than 100 characters
    text = text.replace('_', ' ')
    try:
        score = Textatistic(text).flesch_score
        return score
    except:
        # dealing with no sentence detected case
        try:
            text += '.'
            score = Textatistic(text).flesch_score
            return score
        # dealing with other cases
        except:
            return 0
        
# helper function to get the sentiment score
def sentiment(text):
    sentiments = SentimentIntensityAnalyzer()
    score = sentiments.polarity_scores(text)['compound']
    return score

# helper function to get the number of bad words
def get_bad_words_count(text_lemmatized, bad_words):
    counts = [] 
    for cat in bad_words.Category.unique():
        bad_word_list = list(bad_words[bad_words['Category']==cat].Words)
        count = sum(map(lambda x: text_lemmatized.count(x), bad_word_list))
        counts.append(count)
    return counts

def add_features(data):
    data2 = data.copy()

    # length of text
    #print('creating length of text')
    data2['text_length'] = data2['comment_text'].apply(len)
    
    # sentence
    print('creating sentence features')
    # number of sentences
    #data2['num_sentences'] = data2['POS_tagging'].apply(len)
    # sentence length in words
    sentence_stats = data2['POS_tagging'].apply(get_sentence_stats)
    # average length of sentences in words
    #data2['average_length'] = sentence_stats.apply(lambda x: x[0])
    # maximum length of sentences in words
    #data2['max_length'] = sentence_stats.apply(lambda x: x[1])
    # minimum length of sentences in words
    data2['min_length'] = sentence_stats.apply(lambda x: x[2])
    # standard deviation of sentences length in words
    #data2['std_length'] = sentence_stats.apply(lambda x: x[3])
    
    # words
    print('creating word features')
    # number of words
    data2['num_words'] = data2['text_tokenized_nopunc'].apply(len)
    # number of words vs length
    data2['num_words_vs_length'] = np.where(data2['text_length']>0, data2['num_words']/data2['text_length'], 0)
    # length vs number of words
    #data2['length_vs_num_words'] = np.where(data2['num_words']>0, data2['text_length']/data2['num_words'], 0)
    
    # stop words
    #print('creating stop word features')
    #stop_words = set(stopwords.words("english"))
    # number of stop words
    #data2['num_stop_words'] = data2['text_tokenized_nopunc'].apply(lambda x: len(list(word for word in x if word in stop_words)))
    # number of stop words vs length
    #data2['stop_words_vs_length'] = np.where(data2['text_length']>0, data2['num_stop_words']/data2['text_length'], 0)
    # number of stop words vs words
    #data2['stop_words_vs_words'] = np.where(data2['num_words']>0, data2['num_stop_words']/data2['num_words'], 0)
    
    # exclamation marks
    print('creating exclamation mark features')
    # number of exclamation marks
    data2['num_exclamation_marks'] = data2['noise_removed_text'].apply(lambda x: x.count('!'))
    # number of exclamation marks vs length
    data2['exclamation_marks_vs_length'] = np.where(data2['text_length']>0, data2['num_exclamation_marks']/data2['text_length'], 0)
    # number of exclamation marks vs words
    #data2['exclamation_marks_vs_words'] = np.where(data2['num_words']>0, data2['num_exclamation_marks']/data2['num_words'], 0)
    
    # question marks
    #print('creating question mark features')
    # number of question marks
    #data2['num_question_marks'] = data2['noise_removed_text'].apply(lambda x: x.count('?'))
    # number of question marks vs length
    #data2['question_marks_vs_length'] = np.where(data2['text_length']>0, data2['num_question_marks']/data2['text_length'], 0)
    # number of question marks vs words
    #data2['question_marks_vs_words'] = np.where(data2['num_words']>0, data2['num_question_marks']/data2['num_words'], 0)
    
    # punctuations
    #print('creating punctuations features')
    # number of punctuations
    #data2['num_punctuations'] = data2['noise_removed_text'].apply(get_punctuations)
    # number of punctuations vs length
    #data2['punctuations_vs_length'] = np.where(data2['text_length']>0, data2['num_punctuations']/data2['text_length'], 0)
    # number of punctuations vs words
    #data2['punctuations_vs_words'] = np.where(data2['num_words']>0, data2['num_punctuations']/data2['num_words'], 0)
    
    # unique words and repeated words
    print('creating unique and repeated word features')
    unique_repeated = data2['comment_text'].apply(get_unique_repeated)
    # number of unique words
    data2['num_unique_words'] = unique_repeated.apply(lambda x: x[0])
    # number of unique words vs length
    #data2['unique_words_vs_length'] = np.where(data2['text_length']>0, data2['num_unique_words']/data2['text_length'], 0)
    # number of unique words vs words
    #data2['unique_words_vs_words'] = np.where(data2['num_words']>0, data2['num_unique_words']/data2['num_words'], 0)
    # number of repeated words
    #data2['num_repeated_words'] = unique_repeated.apply(lambda x: x[1])
    # number of repeated words vs length
    #data2['repeated_words_vs_length'] = np.where(data2['text_length']>0, data2['num_repeated_words']/data2['text_length'], 0)
    # number of repeated words vs words
    #data2['repeated_words_vs_words'] = np.where(data2['num_words']>0, data2['num_repeated_words']/data2['num_words'], 0)

    # word tag
    print('creating word tag features')
    word_tag = data2['POS_tagging_flat'].apply(get_word_tag_count)
    # number of nouns
    #data2['num_nouns'] = word_tag.apply(lambda x: x[0])
    # number of nouns vs length
    #data2['nouns_vs_length'] = np.where(data2['text_length']>0, data2['num_nouns']/data2['text_length'], 0)
    # number of nouns vs words
    #data2['nouns_vs_words'] = np.where(data2['num_words']>0, data2['num_nouns']/data2['num_words'], 0)
    # number of adjectives
    #data2['num_adjectives'] = word_tag.apply(lambda x: x[1])
    # number of adjectives vs length
    #data2['adjectives_vs_length'] = np.where(data2['text_length']>0, data2['num_adjectives']/data2['text_length'], 0)
    # number of adjectives vs words
    #data2['adjectives_vs_words'] = np.where(data2['num_words']>0, data2['num_adjectives']/data2['num_words'], 0)
    # number of verbs
    data2['num_verbs'] = word_tag.apply(lambda x: x[2])
    # number of verbs vs length
    data2['verbs_vs_length'] = np.where(data2['text_length']>0, data2['num_verbs']/data2['text_length'], 0)
    # number of verbs vs words
    #data2['verbs_vs_words'] = np.where(data2['num_words']>0, data2['num_verbs']/data2['num_words'], 0)
    
    # uppercase
    print('creating uppercase features')
    uppercase = data2['noise_removed_text'].apply(get_uppercase)
    # number of capitals
    data2['num_uppercase'] = uppercase.apply(lambda x: x[0])
    # number of capitals vs length
    data2['uppercase_vs_length'] = np.where(data2['text_length']>0, data2['num_uppercase']/data2['text_length'], 0)
    # number of capitals vs words
    #data2['uppercase_vs_words'] = np.where(data2['num_words']>0, data2['num_uppercase']/data2['num_words'], 0)
    # number of words that entirely in upper case
    #data2['num_uppercase_words'] = uppercase.apply(lambda x: x[1])
    # number of uppercase words vs length
    #data2['uppercase_words_vs_length'] = np.where(data2['text_length']>0, data2['num_uppercase_words']/data2['text_length'], 0)
    # number of uppercase words vs words
    #data2['uppercase_words_vs_words'] = np.where(data2['num_words']>0, data2['num_uppercase_words']/data2['num_words'], 0)

    # date and time
    #print('creating date and time features')
    # number of timestamp
    #data2["num_timestamp"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}:\d{2}", x))
    # number of dates (long form)
    #data2["num_date_long"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # number of dates (short form)
    #data2["num_date_short"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))

    # IP address, URL, mail address
    #print('creating special sequence features')
    # number of IP address
    #data2["num_ip"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', x))
    # number of http links
    #data2["num_http"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # number of mail address
    #data2["num_mail"] = data2["comment_text"].apply(lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x))
    
    # readability, sentiment, profanity
    print('creating text analysis features')
    # readability score -- time consuming
    #data2["readability"] = data2["noise_removed_text"].apply(readability)
    # sentiment analysis
    data2["sentiment"] = data2["noise_removed_text"].apply(sentiment)
    # profanity check
    
    # bad words
    print('creating bad word features')
    bad_words = pd.read_csv('Data/frequent_bad_words.csv')
    bad_words_count = data2["lemmatization"].apply(lambda x: get_bad_words_count(x, bad_words))
    # number of bad words of toxic comments
    data2['num_bad_toxic'] = bad_words_count.apply(lambda x: x[0])
    # number of bad words of toxic comments vs length
    data2['bad_toxic_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_toxic']/data2['text_length'], 0)
    # number of bad words of toxic comments vs words
    #data2['bad_toxic_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_toxic']/data2['num_words'], 0)
    # number of bad words of severe_toxic comments
    data2['num_bad_severe_toxic'] = bad_words_count.apply(lambda x: x[1])
    # number of bad words of severe_toxic comments vs length
    data2['bad_severe_toxic_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_severe_toxic']/data2['text_length'], 0)
    # number of bad words of severe_toxic comments vs words
    #data2['bad_severe_toxic_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_severe_toxic']/data2['num_words'], 0)
    # number of bad words of obscene comments
    data2['num_bad_obscene'] = bad_words_count.apply(lambda x: x[2])
    # number of bad words of obscene comments vs length
    data2['bad_obscene_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_obscene']/data2['text_length'], 0)
    # number of bad words of obscene comments vs words
    #data2['bad_obscene_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_obscene']/data2['num_words'], 0)
    # number of bad words of threat comments
    data2['num_bad_threat'] = bad_words_count.apply(lambda x: x[3])
    # number of bad words of threat comments vs length
    data2['bad_threat_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_threat']/data2['text_length'], 0)
    # number of bad words of threat comments vs words
    #data2['bad_threat_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_threat']/data2['num_words'], 0)
    # number of bad words of insult comments
    data2['num_bad_insult'] = bad_words_count.apply(lambda x: x[4])
    # number of bad words of insult comments vs length
    data2['bad_insult_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_insult']/data2['text_length'], 0)
    # number of bad words of insult comments vs words
    #data2['bad_insult_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_insult']/data2['num_words'], 0)
    # number of bad words of identity_hate comments
    data2['num_bad_identity_hate'] = bad_words_count.apply(lambda x: x[5])
    # number of bad words of identity_hate comments vs length
    data2['bad_identity_hate_vs_length'] = np.where(data2['text_length']>0, data2['num_bad_identity_hate']/data2['text_length'], 0)
    # number of bad words of identity_hate comments vs words
    #data2['bad_identity_hate_vs_words'] = np.where(data2['num_words']>0, data2['num_bad_identity_hate']/data2['num_words'], 0)
    
    return data2


'''
word embedding
'''
# define a helper function to calculate the embedding vector of each text
def get_embeddings(vectors, text, k, generate_missing=False):
    # dealing with empty text
    if len(text)<1:
        return np.zeros(k)
    # generate randomized vectors for unseen words if generate_missing is True
    if generate_missing:
        vectorized = [vectors[word][:k] if word in vectors else np.random.rand(k) for word in text]
    # represent unseen words with 0 vector if generate_missing is False
    else:
        vectorized = [vectors[word][:k] if word in vectors else np.zeros(k) for word in text]
    # each text is represented by averaging the vectors of its constituent words
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_glove_embedding(data):    
    # build a GloVe model
    print('building glove model')
    glove_model = gensim.models.KeyedVectors.load_word2vec_format("glove.twitter.27B.200d.txt.word2vec", binary=False, limit=1190000)
    
    # generate embedding vectors of size 200 using tokenized text (stop words kept)
    print('generating embedding vectors')
    embeddings_glove = data["text_tokenized_nopunc"].map(lambda x: get_embeddings(glove_model, x, 200))
    
    return embeddings_glove


'''
Evalution
'''
# define a helper function to have scores of different evaluation methods
def get_evaluation_score(y_true, y_pred, y_pred_prob):
    #accuracy
    print('Accuracy score: ', accuracy_score(y_true, y_pred))
    #precision
    print('Precision score: ', precision_score(y_true, y_pred, average="weighted"))
    #recall
    print('Recall score: ', recall_score(y_true, y_pred, average="weighted"))
    #f1
    print('F1 score: ', f1_score(y_true, y_pred, average="weighted"))
    
    #confusion matrix
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    conf_mat_dict={}

    for label_col in range(len(labels)):
        y_true_label = y_true[labels[label_col]]
        y_pred_label = y_pred[:, label_col].toarray()
        conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
    #log loss
    if type(y_pred_prob) == np.ndarray:
        print('Logarithmic Loss: ', log_loss(y_true, y_pred_prob))
    else:
        print('Logarithmic Loss: ', log_loss(y_true, y_pred_prob.toarray()))
    #mean squared error
    #print('Mean squared error: ', mean_squared_error(y_true, y_pred.toarray()))
    #mean absolute error
    #print('Mean absolute error: ', mean_absolute_error(y_true, y_pred.toarray()))
    # AUC score
    if type(y_pred_prob) == np.ndarray:
        print('ROC AUC score: ', roc_auc_score(y_true, y_pred_prob, average="weighted"))
    else:
        print('ROC AUC score: ', roc_auc_score(y_true, y_pred_prob.toarray(), average="weighted"))

# define helper function to get cross valuation score
def get_cross_val_score(model, x_data, y_data, score):
    print('Cross Validation score: ', cross_val_score(model, x_data, y_data, scoring=score, cv=5))

# define a helper function to have evaluation scores of single class models
def get_evaluation_score_single_class(y_true, y_pred, y_pred_prob):
    #accuracy
    print('Accuracy score: ', accuracy_score(y_true, y_pred))
    #precision
    print('Precision score: ', precision_score(y_true, y_pred))
    #recall
    print('Recall score: ', recall_score(y_true, y_pred))
    #f1
    print('F1 score: ', f1_score(y_true, y_pred))
    
    #confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))
    
    #log loss
    print('Logarithmic Loss: ', log_loss(y_true, y_pred_prob))
    
    # AUC score
    print('ROC AUC score: ', roc_auc_score(y_true, y_pred_prob))

# define a helper function to have evaluation scores of single class models write in output file
def get_evaluation_score_single_class_print(y_true, y_pred, y_pred_prob,f):
    #accuracy
    print('Accuracy score: ', accuracy_score(y_true, y_pred), file=f)
    #precision
    print('Precision score: ', precision_score(y_true, y_pred), file=f)
    #recall
    print('Recall score: ', recall_score(y_true, y_pred), file=f)
    #f1
    print('F1 score: ', f1_score(y_true, y_pred), file=f)
    
    #confusion matrix
    print("Confusion matrix:",file=f)
    print(confusion_matrix(y_pred=y_pred, y_true=y_true), file=f)
    
    #log loss
    print('Logarithmic Loss: ', log_loss(y_true, y_pred_prob), file=f)
    
    # AUC score
    print('ROC AUC score: ', roc_auc_score(y_true, y_pred_prob), file=f)

# define a helper function to have scores of overall accuracy, precision, recall and f1
def get_overall_evaluation_score(y_true, y_pred, y_pred_prob):
    #accuracy
    print('Accuracy score: ', accuracy_score(y_true, y_pred))
    #precision
    print('Precision score: ', precision_score(y_true, y_pred, average="weighted"))
    #recall
    print('Recall score: ', recall_score(y_true, y_pred, average="weighted"))
    #f1
    print('F1 score: ', f1_score(y_true, y_pred, average="weighted"))
    #log loss
    print('Logarithmic Loss: ', log_loss(y_true, y_pred_prob))
    
    # AUC score
    print('ROC AUC score: ', roc_auc_score(y_true, y_pred_prob))
    
# define a helper function for KFold validation
def model_evaluation(model, train):
    features = train.columns[7:]
    labels = train.columns[:6]
    kf = KFold(n_splits=5)
    validation_scores = pd.DataFrame({'accuracy':[], 'precision_weighted':[], 'recall_weighted':[], 
                                      'f1_weighted':[], 'log_loss':[], 'roc_auc_weighted':[]})
    for train_index, test_index in kf.split(train):
        train_split = train.iloc[train_index]
        test_split = train.iloc[test_index]
        print('Starting fitting...')
        model.fit(train_split[features], train_split[labels])
        predictions = model.predict(test_split[features])
        predictions_proba = model.predict_proba(test_split[features])
        try:
            roc_auc = roc_auc_score(test_split[labels], predictions_proba.toarray(), average="weighted")
        except:
            roc_auc = 0
        scores = [accuracy_score(test_split[labels], predictions), precision_score(test_split[labels], predictions, average="weighted"),
                  recall_score(test_split[labels], predictions, average="weighted"), 
                  f1_score(test_split[labels], predictions, average="weighted"),
                  log_loss(test_split[labels], predictions_proba.toarray()), roc_auc]
        validation_scores.loc[len(validation_scores)] = scores
        print(f'Evaluation Scores:\n{scores}\n')
    return validation_scores   