import pandas as pd


def findKMostRecentTimestamps(userSortedTimestamp, timestamp, sequence_length):
    length = len(userSortedTimestamp)
    userSortedTimestamp_ts = [0]*sequence_length + userSortedTimestamp.timestamp.tolist()
    userSortedTimestamp_iid = [0]*sequence_length + userSortedTimestamp.item_id.tolist()
    left = sequence_length
    right = length + sequence_length - 1
    mid = (left + right) / 2

    while True:
        mid = int(max(sequence_length, mid))
        
        if userSortedTimestamp_ts[mid] == timestamp: # mid == timestamp
            for i in range(mid, -1, -1):
                if userSortedTimestamp_ts[i] != timestamp:
                    return userSortedTimestamp_ts[i+1-sequence_length : i+1], userSortedTimestamp_iid[i+1-sequence_length : i+1]
        elif mid == length+sequence_length-1: # mid == end
            return userSortedTimestamp_ts[mid-sequence_length : mid], userSortedTimestamp_iid[mid-sequence_length : mid]
        elif userSortedTimestamp_ts[mid] > timestamp and userSortedTimestamp_ts[mid-1] < timestamp: # mid-1 < now < mid
            return userSortedTimestamp_ts[mid-sequence_length : mid], userSortedTimestamp_iid[mid-sequence_length : mid]
        elif userSortedTimestamp_ts[mid+1] > timestamp and userSortedTimestamp_ts[mid] < timestamp: # mid < now < mid+1
            return userSortedTimestamp_ts[mid-sequence_length+1 : mid+1], userSortedTimestamp_iid[mid-sequence_length+1 : mid+1]
        elif userSortedTimestamp_ts[mid] < timestamp:
            left = mid + 1
            mid = (left + right) / 2
        elif userSortedTimestamp_ts[mid] > timestamp:
            right = mid - 1
            mid = (left + right) / 2
        if right < left:
            print "Error"
            exit()

def timestamp_processor(dataset, userSortedTimestamp, sequence_length):
    datetime = pd.to_datetime(dataset.timestamp, unit='s')
    pydatetime = pd.DataFrame(list(map(lambda x: [x.year, x.month-1, x.day-1, x.hour], datetime.dt.to_pydatetime())), columns=['year', 'month', 'date', 'hour'])
    day_of_week = datetime.dt.dayofweek
    day_of_week.name = 'day_of_week'

    dataset = dataset.join(pydatetime)
    dataset = dataset.join(day_of_week)

    KRecentTSItem = list(map(lambda x: findKMostRecentTimestamps(userSortedTimestamp[x[0]], x[1], sequence_length), zip(dataset.user_id, dataset.timestamp)))
    KRecentTimestamps = pd.DataFrame(list(map(lambda x: x[0], KRecentTSItem)), columns=list(map(lambda x: 'timestamp' + str(x), range(sequence_length))))
    KRecentItems = pd.DataFrame(list(map(lambda x: x[1], KRecentTSItem)), columns=list(map(lambda x: 'item_id' + str(x), range(sequence_length))))
    
    dataset = dataset.join(KRecentTimestamps)
    dataset = dataset.join(KRecentItems)

    for i in range(sequence_length):
        datetime = pd.to_datetime(KRecentTimestamps['timestamp' + str(i)], unit='s')
        pydatetime = pd.DataFrame(list(map(lambda x: [x.month-1, x.day-1, x.hour], datetime.dt.to_pydatetime())), columns=['month' + str(i), 'date' + str(i), 'hour' + str(i)])

        day_of_week = datetime.dt.dayofweek
        day_of_week.name = 'day_of_week' + str(i)

        dataset = dataset.join(pydatetime)    
        dataset = dataset.join(day_of_week)
    return dataset