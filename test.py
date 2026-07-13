import os
import torch

# 만약 이 값이 [0] 등으로 되어있으면 1장만 인식합니다. 3장 다 쓰려면 주석 처리하거나 설정하지 않아야 합니다.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" 

print("사용 가능한 GPU 개수:", torch.cuda.device_count())