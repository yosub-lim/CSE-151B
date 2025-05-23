import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 (경로를 정확하게 지정해주세요)
data_path = r"C:\Users\yosub\Downloads\cse151b-spring2025-competition\processed_data_cse151b_v2_corrupted_ssp245\processed_data_cse151b_v2_corrupted_ssp245.zarr"
data = xr.open_zarr(data_path)

# ------------------------------
# 1. 데이터 크기와 차원 확인
# ------------------------------
print("Dataset dimensions:")
print(data.dims)

train_size = len(data.time)  # 전체 시간 축 크기
spatial_dims = (len(data.latitude), len(data.longitude))
print(f"\nTrain/Test time size: {train_size}")
print(f"Spatial dimensions: {spatial_dims} (lat, lon)")

# ------------------------------
# 2. 타겟 변수 분포 (tas, pr)
# ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
data.tas.mean(dim=["x", "y", "member_id"]).sel(ssp="ssp370").plot(ax=axs[0])
axs[0].set_title("Global Mean Temperature (tas) over Time")

data.pr.mean(dim=["x", "y", "member_id"]).sel(ssp="ssp370").plot(ax=axs[1])
axs[1].set_title("Global Mean Precipitation (pr) over Time")
plt.tight_layout()
plt.show()

# ------------------------------
# 3. 입력 변수 분포 시각화 (mean, std)
# ------------------------------
for var in ["CO2", "CH4", "SO2", "BC"]:
    if var in data:
        mean_val = data[var].mean().values
        std_val = data[var].std().values
        print(f"{var} -> mean: {mean_val:.4f}, std: {std_val:.4f}")

# ------------------------------
# 4. 시나리오 및 연도에 따른 분포 변화
# ------------------------------
global_mean = data.sel(ssp=["ssp126", "ssp370", "ssp585"]).mean(dim=["x", "y", "member_id"])

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
global_mean.tas.plot(ax=ax[0], hue="ssp")
ax[0].set_title("Temperature over Time by SSP")

global_mean.pr.plot(ax=ax[1], hue="ssp")
ax[1].set_title("Precipitation over Time by SSP")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. 시간에 따른 기후 강제 변수 (forcings)
# ------------------------------
forcing_mean = data.sel(ssp=["ssp126", "ssp370", "ssp585"]).mean(dim=["longitude", "latitude", "member_id"])
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

forcing_mean.CO2.plot(ax=axs[0], hue="ssp")
axs[0].set_title("CO2 over Time")

forcing_mean.CH4.plot(ax=axs[1], hue="ssp")
axs[1].set_title("CH4 over Time")

forcing_mean.SO2.plot(ax=axs[2], hue="ssp")
axs[2].set_title("SO2 over Time")

forcing_mean.BC.plot(ax=axs[3], hue="ssp")
axs[3].set_title("BC over Time")

plt.tight_layout()
plt.show()

# ------------------------------
# 6. 스냅샷 이미지 출력
# ------------------------------
snapshot = data.isel(time=[0, data.time.size // 2, -1]).sel(ssp="ssp370", member_id=0)

snapshot.tas.plot(col="time", x="x", y="y", robust=True, figsize=(15, 4))
plt.suptitle("Snapshots of Surface Temperature (tas)", y=1.05)
plt.show()

snapshot.pr.plot(col="time", x="x", y="y", robust=True, figsize=(15, 4))
plt.suptitle("Snapshots of Precipitation (pr)", y=1.05)
plt.show()
