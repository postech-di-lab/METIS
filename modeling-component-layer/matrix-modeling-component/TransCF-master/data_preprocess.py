import numpy as np
import pdb
import gzip
import matplotlib
import matplotlib.pyplot as plt
import cPickle as pkl
import operator
import scipy.io as sio
import os.path
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

np.random.seed(23254)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
        
def getuserCache(df):
    userCache = {}
    for uid in sorted(df.uid.unique().tolist()):
        items = sorted(df.loc[df.uid == uid]['iid'].values.tolist())
        userCache[uid] = items

    return userCache
    
def getitemCache(df):
    itemCache = {}
    for iid in sorted(df.iid.unique().tolist()):
        users = sorted(df.loc[df.iid == iid]['uid'].values.tolist())
        itemCache[iid] = users

    return itemCache

def readData(dataset):
    totalFile = pd.read_csv('data/'+dataset+'/ratings.dat',sep="\t",usecols=[0,1],names=['uid','iid'],header=0)
    total_uids = sorted(totalFile.uid.unique())
    total_iids = sorted(totalFile.iid.unique())

    trainFile = pd.read_csv('data/'+dataset+'/LOOTrain.dat',sep="\t",usecols=[0,1],names=['uid','iid'],header=0)
    train_uids = sorted(trainFile.uid.unique())
    train_iids = sorted(trainFile.iid.unique())

    userCache = getuserCache(trainFile)
    itemCache = getitemCache(trainFile)

    root = "data/"+dataset  
    # Read data
    df_data = pd.read_csv(root+'/u.data',sep="\t",names=['uid','iid','rating'])
    df_data = df_data.drop_duplicates(['uid','iid']).reset_index(drop=True)


    # Remove users and items less than 5 ratings
    sr_uid = df_data['uid'].value_counts()
    df_count_uid = pd.DataFrame({'uid': sr_uid.index, 'count': sr_uid.values})
    filtered_uid = df_count_uid.loc[(5 <= df_count_uid['count'])].uid
    df_data = df_data.loc[df_data['uid'].isin(filtered_uid)].reset_index(drop=True)

    sr_iid = df_data['iid'].value_counts()
    df_count_iid = pd.DataFrame({'iid': sr_iid.index, 'count': sr_iid.values})
    filtered_iid = df_count_iid.loc[(5 <= df_count_iid['count'])].iid
    df_data = df_data.loc[df_data['iid'].isin(filtered_iid)].reset_index(drop=True)

    # Remove users with less than 3 ratings to ensure that train/val/test has at least one record each.
    sr_uid = df_data['uid'].value_counts()
    df_count_uid = pd.DataFrame({'uid': sr_uid.index, 'count': sr_uid.values})
    filtered_uid = df_count_uid.loc[(3 <= df_count_uid['count'])].uid
    df_data = df_data.loc[df_data['uid'].isin(filtered_uid)].reset_index(drop=True)

    # map uids and iids from index 0
    unique_uids = df_data.uid.unique()
    unique_iids = df_data.iid.unique()

    uid_map = dict()
    for idx, uid in enumerate(unique_uids):
        uid_map[uid] = idx    

    iid_map = dict()
    for idx, iid in enumerate(unique_iids):
        iid_map[iid] = idx

    return df_data, total_uids, total_iids, train_uids, train_iids, userCache, itemCache, uid_map, iid_map
        
def extractCategory(dataset, df_data, iid_map):
    if dataset == 'ciao':
        cat_name = dict()
        with open("data/"+dataset+"/catalog_ciao.txt") as f:
            for line in f.readlines():
                tmp = line.rstrip().split("\\t")
                original_cat = int(tmp[0])
                name = tmp[1]
                cat_name[original_cat] = name
        
        df_tmp = pd.DataFrame(sio.loadmat('data/'+dataset+'/rating.mat')['rating'],columns=['uid','iid','category','rating','hepfulness'])
        iids = set(df_data.iid.unique())
        meta_dict = {}
        for iid in iids:
            original_cat = sorted(df_tmp.loc[df_tmp.iid == iid]['category'].unique().tolist())[0]
            meta_dict[iid] = cat_name[original_cat]

        category_set = set(df_tmp.category.unique().tolist())
        category_map = dict() 
        for idx, cat in enumerate(category_set):
            category_map[cat_name[cat]] = idx
        
        
    elif dataset == 'cellphone':
        iids = set(df_data.iid.unique())
        meta_dict = {}
        category_set = set()
        cnt = 0
        for l in parse('data/'+dataset+'/meta_Cell_Phones_and_Accessories.json.gz'):
            iid = l['asin']
            if iid in iids:
                category = l['categories'][0]
                if len(category) <= 2 or category[0] != 'Cell Phones & Accessories':
                    meta_dict[iid] = 'N/A'
                    category_set.add('N/A')
                    continue
                meta_dict[iid] = category[2]
                category_set.add(category[2])
    
        category_map = dict()        
        for idx, cat in enumerate(list(category_set)):
            category_map[cat] = idx

    category_map_inv = {v: k for k, v in category_map.iteritems()}
    cat_map = dict()
    mappedIid_mappedCat = dict()
    for original_iid, original_cat in meta_dict.iteritems():
        mapped_cat = category_map[original_cat]
        cat_map[original_iid] = mapped_cat
        mappedIid_mappedCat[iid_map[original_iid]] = mapped_cat
    
    return df_data, cat_map, category_map_inv, mappedIid_mappedCat, category_map

def getEmbeddings(dataset, df_data, total_uids, total_iids, train_uids, train_iids, userCache, itemCache):
    # Get user/item embeddings
    ## CML
    userEmbedding_CML = pkl.load(open('model/userEmbedding_CML_'+dataset+'.pkl'))
    itemEmbedding_CML = pkl.load(open('model/itemEmbedding_CML_'+dataset+'.pkl'))

    ## TransCF
    userEmbedding_TransCF = pkl.load(open('model/userEmbedding_TransCF_'+dataset+'.pkl'))
    itemEmbedding_TransCF = pkl.load(open('model/itemEmbedding_TransCF_'+dataset+'.pkl'))

    userNeighborEmbedding_TransCF = np.zeros((len(total_uids),128))
    for uid in train_uids:
        neighborItems = userCache[uid]
        neighborItems_embeddings = np.mean(itemEmbedding_TransCF[neighborItems],axis=0).tolist()
        userNeighborEmbedding_TransCF[uid,:] = neighborItems_embeddings


    itemNeighborEmbedding_TransCF = np.zeros((len(total_iids),128))
    for iid in train_iids:
        neighborUsers = itemCache[iid]
        neighborUsers_embeddings = np.mean(userEmbedding_TransCF[neighborUsers],axis=0).tolist()
        itemNeighborEmbedding_TransCF[iid,:] = neighborUsers_embeddings

    # Make translation vectors for CML, TransCF_emb, TransCF
    translation_vecs_TransCF = []
    translation_vecs_CML = []
    translation_vecs_TransCF_emb = []
    ratings = []

    categories = []
    for uid in train_uids:
        iids = userCache[uid]
        vec = userNeighborEmbedding_TransCF[uid]
        vec_CML = userEmbedding_CML[uid]
        vec_TransCF = userEmbedding_TransCF[uid]
        tmp_rating = df_data.loc[df_data.uid == uid][:-2]['rating'].values.tolist()
        tmp_category = df_data.loc[df_data.uid == uid][:-2]['cat'].values.tolist()
        categories += tmp_category

        ratings += tmp_rating
        for iid in iids:
            translation = vec * itemNeighborEmbedding_TransCF[iid]
            translation_vecs_TransCF.append(translation)

            translation = vec_CML - itemEmbedding_CML[iid]
            translation_vecs_CML.append(translation)

            translation = vec_TransCF - itemEmbedding_TransCF[iid]
            translation_vecs_TransCF_emb.append(translation)

    translation_vecs_TransCF = np.array(translation_vecs_TransCF)
    translation_vecs_CML = np.array(translation_vecs_CML)
    translation_vecs_TransCF_emb = np.array(translation_vecs_TransCF_emb)
    ratings = np.array(ratings)

    translation_vecs_TransCF_categories = translation_vecs_TransCF
    translation_vecs_CML_categories = translation_vecs_CML
    translation_vecs_TransCF_emb_categories = translation_vecs_TransCF_emb
    categories = np.array(categories)
    
    return itemEmbedding_CML, itemEmbedding_TransCF, translation_vecs_TransCF, translation_vecs_CML, translation_vecs_TransCF_emb, ratings, translation_vecs_TransCF_categories, translation_vecs_CML_categories, translation_vecs_TransCF_emb_categories, categories

def preprocessRatings(translation_vecs_TransCF, translation_vecs_CML, translation_vecs_TransCF_emb, ratings):
    # Preprocess ratings
    translation_vecs_TransCF = translation_vecs_TransCF[(ratings != 0)]
    translation_vecs_CML = translation_vecs_CML[(ratings != 0)]
    translation_vecs_TransCF_emb = translation_vecs_TransCF_emb[(ratings != 0)]
    ratings = ratings[(ratings != 0)]
    
    return translation_vecs_TransCF, translation_vecs_CML, translation_vecs_TransCF_emb, ratings

def preprocessCategories(dataset, itemEmbedding_CML, itemEmbedding_TransCF, translation_vecs_TransCF_categories, translation_vecs_CML_categories, translation_vecs_TransCF_emb_categories, categories, category_map_inv, mappedIid_mappedCat, category_map):
    # Preprocess categories    
    ## Translation
    unique, counts = np.unique(categories, return_counts=True)
    cat_cnt = dict(zip(unique, counts))
    sorted_cat_cnt = sorted(cat_cnt.items(), key=operator.itemgetter(1))
    sorted_cat_cnt.reverse()
    
    if dataset == 'cellphone':
        for idx, elem in enumerate(sorted_cat_cnt):
            if elem[0] == category_map['N/A']:
                break
        sorted_cat_cnt.append(sorted_cat_cnt[idx])
        del sorted_cat_cnt[idx]
    
    top10_cat = [elem[0] for elem in sorted_cat_cnt[:10]]
    rest_cat = [elem[0] for elem in sorted_cat_cnt[10:]]

    for cat in rest_cat:
        translation_vecs_TransCF_categories = translation_vecs_TransCF_categories[(categories != cat)]
        translation_vecs_CML_categories = translation_vecs_CML_categories[(categories != cat)]
        translation_vecs_TransCF_emb_categories = translation_vecs_TransCF_emb_categories[(categories != cat)]
        categories = categories[(categories != cat)]


    originalcategories = []
    for lab in top10_cat:
        originalcategories.append(category_map_inv[lab])
    print("- Top-10 Categories Trans: %s" %(str(originalcategories)))


    ## Embedding
    categories_emb = []
    for iid in range(len(itemEmbedding_CML)):
        categories_emb.append(mappedIid_mappedCat[iid])
    categories_emb = np.array(categories_emb)


    unique, counts = np.unique(categories_emb, return_counts=True)
    cat_cnt = dict(zip(unique, counts))
    sorted_cat_cnt = sorted(cat_cnt.items(), key=operator.itemgetter(1))
    sorted_cat_cnt.reverse()
    
    if dataset == 'cellphone':
        for idx, elem in enumerate(sorted_cat_cnt):
            if elem[0] == category_map['N/A']:
                break
        sorted_cat_cnt.append(sorted_cat_cnt[idx])
        del sorted_cat_cnt[idx]
    
    top10_cat = [elem[0] for elem in sorted_cat_cnt[:10]]
    rest_cat = [elem[0] for elem in sorted_cat_cnt[10:]]

    for cat in rest_cat:
        itemEmbedding_CML = itemEmbedding_CML[(categories_emb != cat)]
        itemEmbedding_TransCF = itemEmbedding_TransCF[(categories_emb != cat)]
        categories_emb = categories_emb[(categories_emb != cat)]

    originalcategories = []
    for lab in top10_cat:
        originalcategories.append(category_map_inv[lab])
    print("- Top-10 Categories Emb: %s" %(str(originalcategories)))
    
    return translation_vecs_TransCF_categories, translation_vecs_CML_categories, translation_vecs_TransCF_emb_categories, categories, itemEmbedding_CML, itemEmbedding_TransCF, categories_emb
    
def balanceData(translation_vecs_TransCF, translation_vecs_CML, translation_vecs_TransCF_emb, ratings, cat=True):    
    # To remove the class imbalance problem for translation vectors
    min_ = 10000000
    for elem in np.unique(ratings):
        numbers = np.sum(ratings == elem)
        if numbers < min_:
            min_ = np.sum(ratings == elem)
    
    translation_vecs_TransCF_balanced = []
    if len(translation_vecs_CML) != 0:
        translation_vecs_CML_balanced = []
        translation_vecs_TransCF_emb_balanced = []
    ratings_balanced = []
    for elem in np.unique(ratings):
        idxs = np.random.choice(np.where((ratings == elem))[0], min_)
        translation_vecs_TransCF_balanced += translation_vecs_TransCF[idxs].tolist()
        if len(translation_vecs_CML) != 0:
            translation_vecs_CML_balanced += translation_vecs_CML[idxs].tolist()
            translation_vecs_TransCF_emb_balanced += translation_vecs_TransCF_emb[idxs].tolist()
        ratings_balanced += ratings[idxs].tolist()
    
    translation_vecs_TransCF = np.array(translation_vecs_TransCF_balanced)
    if len(translation_vecs_CML) != 0:
        translation_vecs_CML = np.array(translation_vecs_CML_balanced)
        translation_vecs_TransCF_emb = np.array(translation_vecs_TransCF_emb_balanced)
    ratings = np.array(ratings_balanced)
    
    return translation_vecs_TransCF, translation_vecs_CML, translation_vecs_TransCF_emb, ratings


def balanceData_emb(itemEmbedding_TransCF, itemEmbedding_CML, categories_emb):    
    # To remove the class imbalance problem for item embeddings
    min_ = 10000000
    for elem in np.unique(categories_emb):
        numbers = np.sum(categories_emb == elem)
        if numbers < min_:
            min_ = np.sum(categories_emb == elem)

    itemEmbedding_CML_balanced = []
    itemEmbedding_TransCF_balanced = []
    categories_emb_balanced = []
    for elem in np.unique(categories_emb):
        idxs = np.random.choice(np.where((categories_emb == elem))[0], min_)
        itemEmbedding_CML_balanced += itemEmbedding_CML[idxs].tolist()
        itemEmbedding_TransCF_balanced += itemEmbedding_TransCF[idxs].tolist()
        categories_emb_balanced += categories_emb[idxs].tolist()

    itemEmbedding_CML = np.array(itemEmbedding_CML_balanced)
    itemEmbedding_TransCF = np.array(itemEmbedding_TransCF_balanced)
    categories_emb = np.array(categories_emb_balanced)
    
    return itemEmbedding_TransCF, itemEmbedding_CML, categories_emb