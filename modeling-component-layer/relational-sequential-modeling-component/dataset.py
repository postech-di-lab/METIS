def movielens_preprocess():
	total_data = {} # {user_id: [[interaction item_id, timestamp], [], ...]}
	item_cnt = {} # {item_id: count_int}
    # ratings.txt -> user_id :: item_id :: rating :: timestamp
	with open('ratings.txt', 'r') as f:
		while True:
			line = f.readline().strip()
			if not line: break
			tokens = line.split('::')
			uid = int(tokens[0]) # user_id
			iid = int(tokens[1]) # item_id
			if not iid in item_cnt:
				item_cnt[iid] = 0
			item_cnt[iid] += 1
			timestamp = int(tokens[3])

			if not uid in total_data:
				total_data[uid] = []
			total_data[uid].append([iid, timestamp])


	tr_data = []
	va_data = []
	te_data = []

	for uid in total_data:
		temp = sorted(total_data[uid], key=lambda x: x[1]) # 개별 user_id의 interaction item_id 를 timestamp 순으로 정렬
		iids = list(map(lambda x: x[0], temp)) # interaction item_id 만 추출 후 list로 가공
		iids = list(filter(lambda x: item_cnt[x] >= 20, iids)) # 전체 데이터에서 20회 이상 등장한 item_id만 사용
		if len(iids) < 3:
			continue # 시퀀스 길이기 3초과일 경우만 사용
        # Leave One Out setting
		te_data.append(iids) 
		va_data.append(iids[:-1])
		tr_data.append(iids[:-2])

	return tr_data, va_data, te_data