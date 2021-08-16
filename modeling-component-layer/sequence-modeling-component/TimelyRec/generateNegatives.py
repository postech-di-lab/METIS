import pandas as pd
import numpy as np

def getNegativeSamples(tr_dataset, userUninteractedItems, userUninteractedTimes, num_user, num_items):
    dataset = tr_dataset.groupby('user_id')

    final_neg_item_data = pd.DataFrame()
    final_neg_time_data = pd.DataFrame()
    final_neg_item_time_data = pd.DataFrame()
    
    for uid, user_data in dataset:
        uninteracted_items = userUninteractedItems[uid]
        uninteracted_times = userUninteractedTimes[uid]
        
        neg_times1 = np.random.choice(uninteracted_times, size=len(user_data)).tolist()
        neg_times2 = np.random.choice(uninteracted_times, size=len(user_data)).tolist()
        neg_items1 = np.random.choice(uninteracted_items, size=len(user_data)).tolist()
        neg_items2 = np.random.choice(uninteracted_items, size=len(user_data)).tolist()
        
        neg_i = user_data.copy()
        neg_t = user_data.copy()
        neg_it = user_data.copy()

        # neg item
        neg_i['item_id'] = neg_items1

        # neg time
        neg_t['timestamp_hour'] = neg_times1
        neg_t['timestamp'] = list(map(lambda x: x * 3600, neg_times1))

        # neg item/time
        neg_it['item_id'] = neg_items2
        neg_it['timestamp_hour'] = neg_times2
        neg_it['timestamp'] = list(map(lambda x: x * 3600, neg_times2))

        neg_i['rating'] = 0
        neg_t['rating'] = 0
        neg_it['rating'] = 0

        final_neg_item_data = pd.concat([final_neg_item_data, neg_i])
        final_neg_time_data = pd.concat([final_neg_time_data, neg_t])
        final_neg_item_time_data = pd.concat([final_neg_item_time_data, neg_it])

    
    return final_neg_item_data, final_neg_time_data, final_neg_item_time_data
