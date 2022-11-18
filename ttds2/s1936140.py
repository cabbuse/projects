import math

import pandas as pd
import numpy as np


def Evaluation(system_results, qrels):
    ir_eval = pd.DataFrame(
        columns=["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"])
    systemNo = max(system_results["system_number"]) + 1
    for num in range(1, systemNo):
        subSysResults = system_results[system_results["system_number"] == num]
        qr_dic = dict(system_results[system_results["system_number"] == num].query_number.value_counts())
        ir_eval["query_number"] = np.array(range(1, 11))
        ir_eval["system_number"] = np.ones((10, 1)) * num
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
            ap = average_precision(QueriesRelDocs,subSysResults,i,start_of_q)

            ##ndcg10
            dcg_at_10 =  dcg_k(10,qrels,subSysResults,i,QueriesRelDocs)
            idcg_10 = idcg_k(10,i,qrels,subSysResults)
            ndcg_10 = dcg_at_10 / idcg_10
            ##ndcg20
            dcg_at_20 = dcg_k(20, qrels, subSysResults, i, QueriesRelDocs)
            idcg_20 = idcg_k(20, i, qrels, subSysResults)
            ndcg_20 = dcg_at_20 / idcg_20
            p = 0

            a = 0

    return


def precisionAtK(subSysResults, start_of_q, k, QueriesRelDocs):
    q_10 = list(subSysResults["doc_number"])[start_of_q:start_of_q + k]
    q_10 = len(set(q_10) & set(QueriesRelDocs)) / k
    return q_10


def recallatK(subSysResults, start_of_q, QueriesRelDocs, k):
    q_50 = list(subSysResults["doc_number"])[start_of_q:start_of_q + k]
    return len(set(q_50) & set(QueriesRelDocs)) / len(QueriesRelDocs)


def R_Precision(subSysResults, start_of_q, QueriesRelDocs, k):
    R = list(subSysResults["doc_number"])[start_of_q:start_of_q + k]
    return len(set(R) & set(QueriesRelDocs)) / len(QueriesRelDocs)

def average_precision(QueriesRelDocs,subSysResults,i,start_of_q):
    apScore = 0
    r = len(QueriesRelDocs)
    n = len(subSysResults[subSysResults["query_number"] == i])
    for ap_s in range(1, n + 1):
        prec = len(
            set(list(subSysResults["doc_number"])[start_of_q:start_of_q + ap_s]) & set(QueriesRelDocs)) / ap_s
        if subSysResults[subSysResults["query_number"] == i]["doc_number"][ap_s - 1] in QueriesRelDocs.tolist():
            apScore += prec
    apScore = apScore / r
    return apScore

def dcg_k(k,qrels,subSysResults,i,QueriesRelDocs):
    ##dcg @ k
    first_doc = qrels[qrels["query_id"] == i]["relevance"].tolist()[0]
    listOfQ = subSysResults[subSysResults["query_number"]==i][:k]
    dcg = 0
    for doc in QueriesRelDocs:
        if doc in list(listOfQ["doc_number"]):
            gain = (qrels[qrels["doc_id"]==doc])#&
            gain = gain[gain["query_id"]==i]["relevance"].tolist()[0]
            G = int(qrels.loc[(qrels["query_id"] == i) & (qrels["doc_id"] == doc)]["relevance"])
            discount = listOfQ[listOfQ["doc_number"]==doc]["rank_of_doc"].tolist()[0]
            if discount==1:
                dcg += gain
            else:
                dcg += gain/ math.log2(discount)
    aaa = 0
    return dcg

def idcg_k(k,i,qrels,subSysResults):
    relevant = list(qrels[qrels["query_id"] == i]["relevance"])
    idcg = relevant[0]
    if len(relevant)<k:
        iterations = len(relevant)
    else:
        iterations = k
    for x in range(1,iterations):
        idcg += relevant[i]/math.log2(x+1)

    return idcg


system_results = pd.read_csv("system_results.csv", header=0, sep=",")
qrels = pd.read_csv("qrels.csv", header=0, sep=",")
ir_eval = Evaluation(system_results, qrels)
