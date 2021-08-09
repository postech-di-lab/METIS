
import numpy as np
import pandas as pd

max_length = 50
def preprocess(data, repetitive, train=False):
    sid = list(data['sessionId'])
    iid = list(data['itemId'])
    data = []

    prev_sid = -1
    for s, i in zip(sid, iid):
        if prev_sid != s:
            data.append([])
        data[-1].append(i + 1)
        prev_sid = s

    data_item = []
    data_length = []
    data_target = []

    for session in data:
        if train:
            if len(session) <= 3: continue
            if len(session) > max_length: 
                sub_session = session[:max_length]
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[1:max_length+1])
            else:
                sub_session = session[:-1]            
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[1:])

        else:
            for ind in range(3, len(session)):
                if not repetitive and session[ind] in session[:ind]:
                    continue
                if ind > max_length:
                    break
                sub_session = session[:ind]
                data_item.append(sub_session)
                data_length.append(float(len(sub_session)))
                data_target.append(session[ind])

    return data_item, data_length, data_target
