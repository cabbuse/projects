import math
import re
from collections import Counter
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from stemming.porter2 import stem
import pandas as pd
import numpy as np


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
        eval.loc[str(len(eval.index) + 1)] = np.around(eval.mean(), 3)
        eval._set_value(str(noQueries), "query_number", "mean")
        # eval.loc["query_number"] = "mean"
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
    aaa = 0
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
    OT = textData.groupby(0).get_group("OT")
    NT = textData.groupby(0).get_group("NT")
    quran = textData.groupby(0).get_group("Quran")
    OT = ' '.join(OT[1].tolist())
    NT = ' '.join(NT[1].tolist())
    quran = ' '.join(quran[1].tolist())
    OT = re.sub(r"[^\w]+", " ", OT).lower().split(" ")
    NT = re.sub(r"[^\w]+", " ", NT).lower().split(" ")
    quran = re.sub(r"[^\w]+", " ", quran).lower().split(" ")
    stopwords = [word.strip("\n") for word in open("stopwords.txt").readlines()]
    OT = [stem(word.lower()) for word in OT if word.isalpha() and word not in stopwords]
    NT = [stem(word.lower()) for word in NT if word.isalpha() and word not in stopwords]
    quran = [stem(word.lower()) for word in quran if word.isalpha() and word not in stopwords]
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
    LDA(NT, OT, quran)

    p = 0

    return


def LDA(NT, OT, Q):
    joinedC = list(NT + OT + Q)
    dictionary = Dictionary([joinedC])
    dictionary.filter_extremes(no_below=50, no_above=0.1)
    corpus = [dictionary.doc2bow(text) for text in [joinedC]]
    lda = LdaModel(corpus, num_topics=20, id2word=dictionary, random_state=1)
    topicD_Q = docTopProb(Q, lda)
    topicD_NT = docTopProb(NT, lda)
    topicD_OT = docTopProb(OT, lda)
    a = 0
    return


def docTopProb(corpus,lda):
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


system_results = pd.read_csv("system_results.csv", header=0, sep=",")
qrels = pd.read_csv("qrels.csv", header=0, sep=",")
# ir_eval = Evaluation(system_results, qrels)

textData = pd.read_csv('train_and_dev.tsv', sep='\t', header=None)
analysis(textData)
