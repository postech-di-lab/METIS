import random
import numpy as np
import torch
import approx_optim
from utils import set_seed
from datetime import datetime

# 사용할 seed 범위
SEEDS = list(range(1, 11))  # 1 ~ 50

def main():
    print("\n\n\n근사 최적화를 통한 전체 연산량 감소 비율 평가\n")

    logs = []
    passed = []  # accel >= 10% 인 seed들 기록

    for i, seed in enumerate(SEEDS):
        # 재현성을 위해 (필요 없으면 주석 처리 가능)
        set_seed(seed)
        print(f"=== Seed {seed} 실행 ({i + 1}/{len(SEEDS)}) ===")
        result_app, result_acc= approx_optim.main(seed)
        # start_time= datetime.fromtimestamp(start_time)
        # end_time=datetime.fromtimestamp(end_time)
        # start_time2= datetime.fromtimestamp(start_time2)
        # end_time2=datetime.fromtimestamp(end_time2)        
        base_training_time, approx_training_time = result_app

        accel = (1 - (approx_training_time / base_training_time)) * 100
        print(accel,"1")
        approx = np.float64(approx_training_time)
        base = np.float64(base_training_time)
        accel = (1 - approx / base) * 100    
        print(accel,"2")
        
        log_str = (
            f"[Seed{seed:>2}] | \n"
            # f"Before_Start_time: {start_time},\n"
            # f"Before_End_time: {end_time},\n"
            f"Before: {base_training_time} sec,\n"
            # f"After_Start_time: {start_time2},\n"
            # f"After_End_time: {end_time2},\n"

            f"After: {approx_training_time} sec,\n"
            f"Accel: {accel}%,\n"
            f"Accuracy: {result_acc * 100:.2f}%\n"
        )
        print(log_str)
        logs.append(log_str)

        if accel >= 10.0:
            passed.append((seed, accel))

    # 전체 로그 저장
    with open("/home/kai0920/Verification/approx-optim-2025/ApproximationOptimization_Seeds.log", "w", encoding="UTF-8") as f:
        for log_str in logs:
            f.write(log_str + "\n")

        # f.write("\n=== accel >= 10% 통과한 seed 목록 ===\n")
        for seed, accel in passed:
            f.write(f"seed {seed:2d} | accel: {accel}%\n")
    
    print(f"로그 저장위치: /home/kai0920/Verification/approx-optim-2025/ApproximationOptimization_Seeds.log")
        # f.write(f"\nTotal seeds with accel >= 10%: {len(passed)} / {len(SEEDS)}\n")

if __name__ == "__main__":
    main()
