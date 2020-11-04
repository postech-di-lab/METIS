import pandas as pd

df = pd.read_json('data/Movies_and_TV_5.json', lines=True)
df = df[['reviewerID', 'asin', 'overall']]

df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'})
df['user'] = df['user'].astype('category').cat.codes
df['item'] = df['item'].astype('category').cat.codes
df.to_json('data/amt.json')

df = pd.read_json('data/amt.json')
for i in range(25):
    df_gb = df.groupby('user').count()
    users = df_gb[df_gb['item'] > 25].index
    df = df[df['user'].isin(users)]

    df_gb = df.groupby('item').count()
    items = df_gb[df_gb['user'] > 25].index
    df = df[df['item'].isin(items)]

df['user'] = df['user'].astype('category').cat.codes
df['item'] = df['item'].astype('category').cat.codes
df.to_json('data/amt_preprocessed.json')

df = pd.read_json('data/amt_preprocessed.json')
df = df.sample(frac=1, random_state=0)

split = int(len(df) * 0.8)
df[:split].to_json('data/amt_train.json')
df[split:].to_json('data/amt_test.json')
