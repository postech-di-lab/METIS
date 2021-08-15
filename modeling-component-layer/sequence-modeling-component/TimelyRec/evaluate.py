import pandas as pd
import math

def evaluate(model, va_dataset, num_candidates, sequence_length):
    HR1 = 0
    HR5 = 0
    NDCG5 = 0
    HR10 = 0
    NDCG10 = 0
    num_users = int(len(va_dataset) / num_candidates)

    for i in range(num_users):
        va_batch = va_dataset.iloc[i * num_candidates : (i + 1) * num_candidates]

        user_input = va_batch.user_id
        item_input = va_batch.item_id

        recent_month_inputs = []
        recent_day_inputs = []
        recent_date_inputs = []
        recent_hour_inputs = []
        recent_timestamp_inputs = []
        recent_itemid_inputs = []

        month_input = va_batch.month
        day_input = va_batch.day_of_week
        date_input = va_batch.date
        hour_input = va_batch.hour
        timestamp_input = va_batch.timestamp
        for j in range(sequence_length):
            recent_month_inputs.append(va_batch['month' + str(j)])
            recent_day_inputs.append(va_batch['day_of_week' + str(j)])
            recent_date_inputs.append(va_batch['date' + str(j)])
            recent_hour_inputs.append(va_batch['hour' + str(j)])
            recent_timestamp_inputs.append(va_batch['timestamp' + str(j)])
            recent_itemid_inputs.append(va_batch['item_id' + str(j)])
        labels = va_batch.rating

        prob = pd.DataFrame(model.predict([user_input, item_input, month_input, day_input, date_input, hour_input, timestamp_input] + [recent_month_inputs[j] for j in range(sequence_length)]+ [recent_day_inputs[j] for j in range(sequence_length)]+ [recent_date_inputs[j] for j in range(sequence_length)]+ [recent_hour_inputs[j] for j in range(sequence_length)]+ [recent_timestamp_inputs[j] for j in range(sequence_length)] + [recent_itemid_inputs[j] for j in range(sequence_length)], batch_size=len(va_batch)), columns=['prob'])
        
        va_batch = (va_batch.reset_index(drop=True)).join(prob)        
        top1 = va_batch.nlargest(1, 'prob')
        top5 = va_batch.nlargest(5, 'prob')
        top10 = va_batch.nlargest(10, 'prob')
        hit1 = int(1 in top1.rating.tolist())
        hit5 = int(1 in top5.rating.tolist())
        hit10 = int(1 in top10.rating.tolist())

        if hit1:
            ind1 = top1.rating.tolist().index(1) + 1

        if hit5:
            ind5 = top5.rating.tolist().index(1) + 1
            ndcg5 = float(1) / math.log(float(ind5 + 1), 2)
        else:
            ndcg5 = 0

        if hit10:
            ind10 = top10.rating.tolist().index(1) + 1
            ndcg10 = float(1) / math.log(float(ind10 + 1), 2)
        else:
            ndcg10 = 0

        HR1 += hit1
        HR5 += hit5
        NDCG5 += ndcg5
        HR10 += hit10
        NDCG10 += ndcg10

        va_batch.drop(columns=['prob'], inplace=True)

    HR1 = float(HR1) / num_users
    HR5 = float(HR5) / num_users
    NDCG5 = NDCG5 / num_users
    HR10 = float(HR10) / num_users
    NDCG10 = NDCG10 / num_users

    return HR1, HR5, NDCG5, HR10, NDCG10