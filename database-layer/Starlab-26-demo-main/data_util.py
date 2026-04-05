import pickle

data_path = "./fcrec_data/dataset/ml-100k/TASK_0.pickle"

with open(data_path, "rb") as f:
    task_data = pickle.load(f)

# 1. item_list 확인
item_list = task_data["item_list"]
print(f"전체 아이템 개수 (총 상호작용 수): {len(item_list)}")
print(f"아이템 ID 최솟값: {min(item_list)}")
print(f"아이템 ID 최댓값: {max(item_list)}")

# 2. train_dict 내부 확인
train_dict = task_data["train_dict"]
sample_user = list(train_dict.keys())[0]
sample_items = train_dict[sample_user] # .keys() 제거

print(f"샘플 사용자 ID: {sample_user}")
print(f"해당 사용자의 상호작용 아이템 ID 예시: {sample_items[:5]}")