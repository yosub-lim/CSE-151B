import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
data_path = r"C:\Users\yosub\Downloads\cse151b-spring2025-competition\processed_data_cse151b_v2_corrupted_ssp245\processed_data_cse151b_v2_corrupted_ssp245.zarr"
data = xr.open_zarr(data_path)

# 테스트 SSP 및 멤버 선택
test_data = data.sel(ssp="ssp245", member_id=0)

# 출력 변수 (tas = temperature, pr = precipitation)
target_var = "tas"  # 또는 "pr"
y_true = test_data[target_var].isel(time=slice(-360, None))  # 마지막 360개

# 예측값 로드: 예시로 랜덤 예측 (실제론 모델 출력을 로드해야 함)
# y_pred = load_your_model_predictions()  ← 너가 따로 제공해야 해
# 임시로 랜덤으로 예측 생성 (이거 대신 너의 모델 예측값을 넣어줘!)
y_pred = y_true + np.random.normal(scale=2.0, size=y_true.shape)

# 오류 계산: MSE (공간 평균)
mse = ((y_pred - y_true) ** 2).mean(dim=["x", "y"])
topk_indices = mse.argsort()[-3:]  # 가장 큰 오류 3개

# 시각화
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
for i, idx in enumerate(topk_indices.values):
    true_map = y_true.isel(time=idx)
    pred_map = y_pred.isel(time=idx)
    error_map = (pred_map - true_map)

    true_map.plot(ax=axs[i, 0], cmap="coolwarm")
    axs[i, 0].set_title(f"True [{idx}]")

    pred_map.plot(ax=axs[i, 1], cmap="coolwarm")
    axs[i, 1].set_title(f"Pred [{idx}]")

    error_map.plot(ax=axs[i, 2], cmap="bwr")
    axs[i, 2].set_title(f"Error [{idx}]")

plt.tight_layout()
plt.show()
