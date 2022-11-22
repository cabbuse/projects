import math
import re
from collections import Counter
import sklearn.svm
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from stemming.porter2 import stem
import pandas as pd
import numpy as np
import scipy
import random
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from nltk.tokenize import TweetTokenizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
import nltk


def Evaluation(system_results, qrels):
    ir_eval = pd.DataFrame(
        columns=["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"])
    systemNo = max(system_results["system_number"]) + 1
    for num in range(1, systemNo):
        eval = pd.DataFrame(
            columns=["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"])
        subSysResults = system_results[system_results["system_number"] == num]
        qr_dic = dict(system_results[system_results["system_number"] == num].query_number.value_counts())
        eval["query_number"] = np.array(range(1, 11))
        eval["system_number"] = np.ones((10, 1)) * num
        P_10 = []
        R_50 = []
        r_precision = []
        AP = []
        nDCG_10 = []
        nDCG_20 = []
        start_of_q = 0
        noQueries = max(qrels["query_id"]) + 1
        for i in range(1, noQueries):
            QueriesRelDocs = qrels[qrels["query_id"] == i]["doc_id"]
            q_10 = precisionAtK(subSysResults, start_of_q, 10, QueriesRelDocs)
            q_50 = recallatK(subSysResults, start_of_q, QueriesRelDocs, 50)
            R = R_Precision(subSysResults, start_of_q, QueriesRelDocs, 50)
            # average precision here
            ap = average_precision(QueriesRelDocs, subSysResults, i, start_of_q)

            ##ndcg10
            dcg_at_10 = dcg_k(10, qrels, subSysResults, i, QueriesRelDocs)
            idcg_10 = idcg_k(10, i, qrels, subSysResults)
            ndcg_10 = dcg_at_10 / idcg_10
            ##ndcg20
            dcg_at_20 = dcg_k(20, qrels, subSysResults, i, QueriesRelDocs)
            idcg_20 = idcg_k(20, i, qrels, subSysResults)
            ndcg_20 = dcg_at_20 / idcg_20
            start_of_q += qr_dic[i]
            P_10.append(q_10)
            R_50.append(q_50)
            r_precision.append(R)
            AP.append(ap)
            nDCG_10.append(ndcg_10)
            nDCG_20.append(ndcg_20)
        eval["P@10"] = np.around(np.array(P_10), 3)
        eval["R@50"] = np.around(np.array(R_50), 3)
        eval["r-precision"] = np.around(np.array(r_precision), 3)
        eval["AP"] = np.around(np.array(AP), 3)
        eval["nDCG@10"] = np.around(np.array(nDCG_10), 3)
        eval["nDCG@20"] = np.around(np.array(nDCG_20), 3)
        eval.index = eval.index + 1
        eval.loc["mean"] = eval.mean()
        # concat dataframe
        ir_eval = pd.concat([ir_eval, eval], axis=0)

    return ir_eval


def precisionAtK(subSysResults, start_of_q, k, QueriesRelDocs):
    q_10 = list(subSysResults["doc_number"])[start_of_q:start_of_q + k]
    q_10 = len(set(q_10) & set(QueriesRelDocs)) / k
    return q_10


def recallatK(subSysResults, start_of_q, QueriesRelDocs, k):
    q_50 = list(subSysResults["doc_number"])[start_of_q:start_of_q + k]
    return len(set(q_50) & set(QueriesRelDocs)) / len(QueriesRelDocs)


def R_Precision(subSysResults, start_of_q, QueriesRelDocs, k):
    R = list(subSysResults["doc_number"])[start_of_q:start_of_q + len(QueriesRelDocs)]
    r_prec = len(set(R) & set(QueriesRelDocs)) / len(QueriesRelDocs)
    return r_prec


def average_precision(QueriesRelDocs, subSysResults, i, start_of_q):
    apScore = 0
    r = len(QueriesRelDocs)
    n = len(subSysResults[subSysResults["query_number"] == i])
    for ap_s in range(1, n + 1):
        prec = len(
            set(list(subSysResults["doc_number"])[start_of_q:start_of_q + ap_s]) & set(QueriesRelDocs)) / ap_s
        if (subSysResults[subSysResults["query_number"] == i]["doc_number"]).tolist()[
            ap_s - 1] in QueriesRelDocs.tolist():
            apScore += prec
    apScore = apScore / r
    return apScore


def dcg_k(k, qrels, subSysResults, i, QueriesRelDocs):
    ##dcg @ k
    first_doc = qrels[qrels["query_id"] == i]["relevance"].tolist()[0]
    listOfQ = subSysResults[subSysResults["query_number"] == i][:k]
    dcg = 0
    for doc in QueriesRelDocs:
        if doc in list(listOfQ["doc_number"]):
            gain = (qrels[qrels["doc_id"] == doc])  # &
            gain = gain[gain["query_id"] == i]["relevance"].tolist()[0]
            G = int(qrels.loc[(qrels["query_id"] == i) & (qrels["doc_id"] == doc)]["relevance"])
            discount = listOfQ[listOfQ["doc_number"] == doc]["rank_of_doc"].tolist()[0]
            if discount == 1:
                dcg += gain
            else:
                dcg += gain / math.log2(discount)

    return dcg


def idcg_k(k, i, qrels, subSysResults):
    relevant = list(qrels[qrels["query_id"] == i]["relevance"])
    idcg = relevant[0]
    if len(relevant) < k:
        iterations = len(relevant)
    else:
        iterations = k
    for x in range(1, iterations):
        idcg += relevant[x] / math.log2(x + 1)

    return idcg


def analysis(textData):
    sno = nltk.stem.SnowballStemmer('english')
    stopwords = [word.strip("\n") for word in open("stopwords.txt").readlines()]
    OT = textData.groupby(0).get_group("OT")
    NT = textData.groupby(0).get_group("NT")
    quran = textData.groupby(0).get_group("Quran")

    qurancorpus = [lis.replace("Quran", "") for lis in list(quran[1])]
    OTcorpus = [lis.replace("OT", "") for lis in list(OT[1])]
    NTcorpus = [lis.replace("NT", "") for lis in list(NT[1])]
    OT = re.sub(r"[^\w]+", " ", ' '.join(OTcorpus)).lower().split(" ")
    NT = re.sub(r"[^\w]+", " ", ' '.join(NTcorpus)).lower().split(" ")
    quran = re.sub(r"[^\w]+", " ", ' '.join(qurancorpus)).lower().split(" ")

    qurancorpus = [re.sub(r"[^\w]+", " ", x).lower().split(" ") for x in qurancorpus]
    OTcorpus = [re.sub(r"[^\w]+", " ", x).lower().split(" ") for x in OTcorpus]
    NTcorpus = [re.sub(r"[^\w]+", " ", x).lower().split(" ") for x in NTcorpus]
    qurancorpus = [[[sno.stem(word.lower()) for word in x if word.isalpha() and word not in stopwords]] for x in
                   qurancorpus]
    OTcorpus = [[[sno.stem(word.lower()) for word in x if word.isalpha() and word not in stopwords]] for x in OTcorpus]
    NTcorpus = [[[sno.stem(word.lower()) for word in x if word.isalpha() and word not in stopwords]] for x in NTcorpus]

    OT = [sno.stem(word.lower()) for word in OT if word.isalpha() and word not in stopwords]
    NT = [sno.stem(word.lower()) for word in NT if word.isalpha() and word not in stopwords]
    quran = [sno.stem(word.lower()) for word in quran if word.isalpha() and word not in stopwords]

    N_dict = dict(Counter(NT))
    O_dict = dict(Counter(OT))
    Q_dict = dict(Counter(quran))
    all_dict = dict(Counter(O_dict) + Counter(N_dict) + Counter(Q_dict))
    NMI, NChi = MI_CHI(all_dict, N_dict, NT, OT, quran, "NT")
    OMI, OChi = MI_CHI(all_dict, O_dict, NT, OT, quran, "OT")
    QMI, QChi = MI_CHI(all_dict, Q_dict, NT, OT, quran, "Q")
    print("new testament top 10 MI and chi ")
    print(Counter(NMI).most_common(10))
    print(Counter(NChi).most_common(10))
    print("\n")
    print("old testament top 10 MI and chi")
    print(Counter(OMI).most_common(10))
    print(Counter(OChi).most_common(10))
    print("\n")
    print("quran top 10 MI and chi ")
    print(Counter(QMI).most_common(10))
    print(Counter(QChi).most_common(10))
    print("\n")
    topicD_OT, topicD_NT, topicD_Q, lda = LDA(NTcorpus, OTcorpus, qurancorpus)

    topic_ranked_NT = sorted(topicD_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    topic_ranked_OT = sorted(topicD_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    topic_ranked_Quran = sorted(topicD_Q.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    for topic in topic_ranked_NT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
    for topic in topic_ranked_OT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
    for topic in topic_ranked_Quran:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))

    p = 0

    return


def LDA(NTcorpus, OTcorpus, Qcorpus):
    NTcorpus = [x[0] for x in NTcorpus]
    OTcorpus = [x[0] for x in OTcorpus]
    Qcorpus = [x[0] for x in Qcorpus]
    joinedC = (NTcorpus + OTcorpus + Qcorpus)
    dictionary = Dictionary(joinedC)
    dictionary.filter_extremes(no_below=50, no_above=0.15)
    corpus = [dictionary.doc2bow(text) for text in joinedC]
    lda = LdaModel(corpus, num_topics=20, id2word=dictionary, random_state=1)
    topicD_Q = docTopProb(Qcorpus, lda)
    topicD_NT = docTopProb(NTcorpus, lda)
    topicD_OT = docTopProb(OTcorpus, lda)
    a = 0
    return topicD_OT, topicD_NT, topicD_Q, lda


def docTopProb(corpus, lda):
    dictionary1 = Dictionary(corpus)
    dictionary1.filter_extremes(no_below=50, no_above=0.1)
    corpus1 = [dictionary1.doc2bow(text) for text in corpus]
    topics = lda.get_document_topics(corpus1)
    topic_dic = {}
    for doc in topics:
        for topic in doc:
            if topic[0] not in topic_dic.keys():
                topic_dic[topic[0]] = topic[1]
            else:
                topic_dic[topic[0]] += topic[1]
    for k, v in topic_dic.items():
        topic_dic[k] = v / len(corpus)

    return topic_dic


def MI_CHI(allDict, Cdict, NT, OT, Q, Cname):
    if Cname == "NT":
        other_dic = dict(Counter(OT) + Counter(Q))
        total_1 = len(NT)
        total_2 = len(Q) + len(OT)
    if Cname == "OT":
        other_dic = dict(Counter(NT) + Counter(Q))
        total_1 = len(OT)
        total_2 = len(Q) + len(NT)
    if Cname == "Q":
        other_dic = dict(Counter(OT) + Counter(NT))
        total_1 = len(Q)
        total_2 = len(NT) + len(OT)
    N = total_1 + total_2
    mi_dic = {}
    chi_dic = {}
    for term in allDict.keys():
        if term in Cdict.keys():
            # compute MI score
            N11 = Cdict[term]
            N01 = total_1 - N11
            if term in other_dic.keys():
                N10 = other_dic[term]
                N00 = total_2 - N10
                MI = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) \
                     + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) \
                     + N10 / N * math.log2(float(N * N10) / float((N10 + N11) * (N00 + N10))) \
                     + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
            else:
                N10 = 0
                N00 = total_2
                MI = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) \
                     + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) \
                     + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
        else:
            N11 = 0
            N01 = total_1
            N10 = other_dic[term]
            N00 = total_2 - N10
            MI = N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) \
                 + N10 / N * math.log2(float(N * N10) / float((N10 + N11) * (N00 + N10))) \
                 + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))

        mi_dic[term] = MI
        # compute Chi_square score
        Chi_square = ((N11 + N10 + N01 + N00) * math.pow(N11 * N00 - N10 * N01, 2)) / (
                (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))
        chi_dic[term] = Chi_square

    return mi_dic, chi_dic


def to_BOW_M(processedData, IDdict):
    matrix_size = (len(processedData), len(IDdict) + 1)
    X = scipy.sparse.dok_matrix(matrix_size)
    oov_index = len(IDdict)
    for doc_id, doc in enumerate(processedData):
        for word in doc:
            # add count for the word
            X[doc_id, IDdict.get(word, oov_index)] += 1
    return X


def pre_processTweets(data):
    tweets = list(data["tweet"])
    # tweetToken = [re.sub(r"[^\w]+", " ", x) for x in tweets]
    # tweetToken = [re.sub(
    #     r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
    #     " ", x.lower()) for x in tweets]
    tweetToken = [re.sub(r"(https?:\/\/)?(www\.)?\w+\.\w+ ?", "", x.lower()) for x in tweets]
    tweetToken = [re.findall("[A-Z\-\']{2,}(?![a-z])|[A-Z\-\'][a-z\-\']+(?=[A-Z])|[\'\w\-]+", x) for x in tweetToken]
    allTweets = sum(tweetToken, [])
    index = 0
    allDict = {}
    for term in allTweets:
        if term not in allDict:
            allDict[term] = index
            index += 1
    # sparseMatrix = to_BOW_M(tweetToken,allDict)

    categories = list(data['sentiment'])
    index = 0
    categoryDict = {}
    for term in categories:
        if term not in categoryDict:
            categoryDict[term] = index
            index += 1
    categoryVal = [categoryDict[x] for x in categories]
    return allDict, categoryDict, categoryVal, tweetToken


def pre_processTweets2(data,stopwords):
    tweets = list(data["tweet"])
    # tweetToken = [re.sub(r"[^\w]+", " ", x) for x in tweets]
    tt = TweetTokenizer()
    tweetToken = [tt.tokenize(x) for x in tweets]
    allTweets = sum(tweetToken, [])
    index = 0
    allDict = {}
    for term in allTweets:
        if term not in allDict:
            allDict[term] = index
            index += 1
    # sparseMatrix = to_BOW_M(tweetToken,allDict)

    categories = list(data['sentiment'])
    index = 0
    categoryDict = {}
    for term in categories:
        if term not in categoryDict:
            categoryDict[term] = index
            index += 1
    categoryVal = [categoryDict[x] for x in categories]
    return allDict, categoryDict, categoryVal, tweetToken

def pre_processTweets3(data):
    tweets = list(data["tweet"])
    #tweetToken = [re.sub(
    #    r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
    #    " ", x.lower()) for x in tweets]
    sno = nltk.stem.SnowballStemmer('english')
    tt = TweetTokenizer()
    tweetToken = [tt.tokenize(x) for x in tweets]

    tweetToken = [[sno.stem(x) for x in y] for y in tweetToken]
    allTweets = sum(tweetToken, [])
    index = 0
    allDict = {}
    for term in allTweets:
        if term not in allDict:
            allDict[term] = index
            index += 1
    # sparseMatrix = to_BOW_M(tweetToken,allDict)

    categories = list(data['sentiment'])
    index = 0
    categoryDict = {}
    for term in categories:
        if term not in categoryDict:
            categoryDict[term] = index
            index += 1
    categoryVal = [categoryDict[x] for x in categories]
    return allDict, categoryDict, categoryVal, tweetToken


def computation_results(Prediction, Actual, catDict):
    Apositive = []
    Anegative = []
    Aneutral = []
    Ppositive = []
    Pnegative = []
    Pneutral = []
    Prediction = [catDict[x] for x in Prediction]
    Actual = [catDict[x] for x in Actual]
    for i in range(len(Actual)):
        if Actual[i] == 2:
            Apositive.append(Actual[i])
            Ppositive.append(Prediction[i])
        elif Actual[i] == 1:
            Aneutral.append(Actual[i])
            Pneutral.append(Prediction[i])
        elif Actual[i] == 0:
            Anegative.append(Actual[i])
            Pnegative.append(Prediction[i])

    p_pos = precision_score(Ppositive, Apositive,average="micro")
    r_pos = recall_score(Ppositive, Apositive,average="micro")
    f_pos = f1_score(Ppositive, Apositive,average="micro")
    p_neg = precision_score(Pnegative, Anegative,average="micro")
    r_neg = recall_score(Pnegative, Anegative,average="micro")
    f_neg = f1_score(Pnegative, Anegative,average="micro")
    p_neu = precision_score(Pneutral, Aneutral,average="micro")
    r_neu = recall_score(Pneutral, Aneutral,average="micro")
    f_neu = f1_score(Pneutral, Aneutral,average="micro")
    p_macro = precision_score(Prediction, Actual,average="micro")
    r_macro = recall_score(Prediction, Actual,average="micro")
    f_macro = f1_score(Prediction, Actual,average="micro")
    format_array = [p_pos,r_pos,f_pos,p_neg,r_neg,f_neg,p_neu,r_neu,f_neu,p_macro,r_macro,f_macro]
    return format_array


def Train_Dev_split(preprocessed_data, categories):
    preprocessed_training_data = []
    training_categories = []
    preprocessed_dev_data = []
    dev_categories = []
    random.seed(1)
    # generate random indexes for development set
    dev_index = [random.randint(0, len(preprocessed_data)) for _ in range(len(preprocessed_data) // 10)]
    # get verses of these index for development set
    for i in dev_index:
        preprocessed_dev_data.append(preprocessed_data[i])
        dev_categories.append(categories[i])
    # get verses of other indexes for train set
    train_index = [i for i in range(len(preprocessed_data)) if i not in dev_index]
    for i in train_index:
        preprocessed_training_data.append(preprocessed_data[i])
        training_categories.append(categories[i])
    return preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index

## ir_eval code
system_results = pd.read_csv("system_results.csv", header=0, sep=",")
qrels = pd.read_csv("qrels.csv", header=0, sep=",")
ir_eval = Evaluation(system_results, qrels)
f1 = "ir_eval.csv"
with open(f1, "a+") as file:
    file.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n")
for index, row in ir_eval.iterrows():
    with open(f1, "a+") as file:
        file.write(str(int(row["system_number"])) + "," + str(index) + "," + "{:.3f}".format(row["P@10"]) + \
                    "," "{:.3f}".format(row["R@50"]) + "," + "{:.3f}".format(row["r-precision"]) + \
                    "," "{:.3f}".format(row["AP"]) + "," + "{:.3f}".format(row["nDCG@10"]) + "," + \
                    "{:.3f}".format(row["nDCG@20"]) + "\n")


##analysis code
textData = pd.read_csv('train_and_dev.tsv', sep='\t', header=None)
analysis(textData)





## text classification code
stopwords = [word.strip("\n") for word in open("stopwords.txt").readlines()]
sentimentData = pd.read_csv('sentiment.tsv', sep='\t', header=None)
sentimentData = sentimentData.rename(columns=sentimentData.iloc[0]).drop(sentimentData.index[0])

##baseline
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
model = sklearn.svm.SVC(C=1000)
model.fit(sparsematrix, y_train)
trainDataPredict = model.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = model.predict(to_BOW_M(X_test, allDict))
a1 = computation_results(testDataPredict, y_test ,catDict)

#
# #c= 10
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
model = sklearn.svm.SVC(C=10)
model.fit(sparsematrix, y_train)
trainDataPredict = model.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = model.predict(to_BOW_M(X_test, allDict))
a2 = computation_results(testDataPredict, y_test ,catDict)

#
# #use nltk, stopword removal  + c=10
allDict, catDict, categoryVal, tweet_token = pre_processTweets2(sentimentData,stopwords)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
model = sklearn.svm.SVC(C=1000)
model.fit(sparsematrix, y_train)
trainDataPredict = model.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = model.predict(to_BOW_M(X_test, allDict))
a3 = computation_results(testDataPredict, y_test ,catDict)


#use stemming, nltk + c= 10
allDict, catDict, categoryVal, tweet_token = pre_processTweets3(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
model = sklearn.svm.SVC(C=1000)
model.fit(sparsematrix, y_train)
trainDataPredict = model.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = model.predict(to_BOW_M(X_test, allDict))
a4 = computation_results(testDataPredict, y_test ,catDict)



## use random forest
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(sparsematrix, y_train)
trainDataPredict = classifier.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = classifier.predict(to_BOW_M(X_test, allDict))
a5 = computation_results(testDataPredict, y_test ,catDict)

##logistic regression with onevsrest
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
model = OneVsRestClassifier(LogisticRegression(random_state=0))
model.fit(sparsematrix, y_train)
trainDataPredict = model.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = model.predict(to_BOW_M(X_test, allDict))
a6 = computation_results(testDataPredict, y_test ,catDict)


## decision tree classifier
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(sparsematrix, y_train)
trainDataPredict = classifier.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = classifier.predict(to_BOW_M(X_test, allDict))
a7 = computation_results(testDataPredict, y_test ,catDict)


## k-nearest neighbours classifier
allDict, catDict, categoryVal, tweet_token = pre_processTweets(sentimentData)
X_train, X_test, y_train, y_test = train_test_split(tweet_token, (sentimentData['sentiment']).tolist(), test_size=0.4,random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5,random_state=42)
sparsematrix = to_BOW_M(X_train, allDict)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(sparsematrix, y_train)
trainDataPredict = classifier.predict(to_BOW_M(X_train, allDict))
devDataPredict = classifier.predict(to_BOW_M(X_dev, allDict))
testDataPredict = classifier.predict(to_BOW_M(X_test, allDict))
a8 = computation_results(testDataPredict, y_test ,catDict)


# print macro f1-score
print(a1[11])
print(a2[11])
print(a3[11])
print(a4[11])
print(a5[11])
print(a6[11])
print(a7[11])
print(a8[11])
##

