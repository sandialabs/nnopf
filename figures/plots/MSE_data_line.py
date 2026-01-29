import numpy as np
import matplotlib.pyplot as plt

### PG (CASE 14) ######################################################################
all_pg_MSE_14 = np.load("../all_pg_MSE_array_14_Kplus1.npy")

# verifying mean values match those in the table
pg_MSE_mean_list_14 = []
for mse_list in all_pg_MSE_14:
    pg_MSE_mean_list_14.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_pg_MSE_14 = []
for mse_list in all_pg_MSE_14:
    unnested_all_pg_MSE_14.extend(mse_list)

# obtain quantile values
pg_min_14 = np.min(unnested_all_pg_MSE_14)
pg_q1_14 = np.quantile(unnested_all_pg_MSE_14, 0.25)
pg_median_14 = np.quantile(unnested_all_pg_MSE_14, 0.5)
pg_q3_14 = np.quantile(unnested_all_pg_MSE_14, 0.75)
pg_max_14 = np.max(unnested_all_pg_MSE_14)


### QG (CASE 14) ######################################################################
all_qg_MSE_14 = np.load("../all_qg_MSE_array_14_Kplus1.npy")

# verifying mean values match those in the table
qg_MSE_mean_list_14 = []
for mse_list in all_qg_MSE_14:
    qg_MSE_mean_list_14.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_qg_MSE_14 = []
for mse_list in all_qg_MSE_14:
    unnested_all_qg_MSE_14.extend(mse_list)

# obtain quantile values
qg_min_14 = np.min(unnested_all_qg_MSE_14)
qg_q1_14 = np.quantile(unnested_all_qg_MSE_14, 0.25)
qg_median_14 = np.quantile(unnested_all_qg_MSE_14, 0.5)
qg_q3_14 = np.quantile(unnested_all_qg_MSE_14, 0.75)
qg_max_14 = np.max(unnested_all_qg_MSE_14)


### VA (CASE 14) ######################################################################
all_va_MSE_14 = np.load("../all_va_MSE_array_14_Kplus1.npy")

# verifying mean values match those in the table
va_MSE_mean_list_14 = []
for mse_list in all_va_MSE_14:
    va_MSE_mean_list_14.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_va_MSE_14 = []
for mse_list in all_va_MSE_14:
    unnested_all_va_MSE_14.extend(mse_list)

# obtain quantile values
va_min_14 = np.min(unnested_all_va_MSE_14)
va_q1_14 = np.quantile(unnested_all_va_MSE_14, 0.25)
va_median_14 = np.quantile(unnested_all_va_MSE_14, 0.5)
va_q3_14 = np.quantile(unnested_all_va_MSE_14, 0.75)
va_max_14 = np.max(unnested_all_va_MSE_14)


### VM (CASE 14) ######################################################################
all_vm_MSE_14 = np.load("../all_vm_MSE_array_14_Kplus1.npy")

# verifying mean values match those in the table
vm_MSE_mean_list_14 = []
for mse_list in all_vm_MSE_14:
    vm_MSE_mean_list_14.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_vm_MSE_14 = []
for mse_list in all_vm_MSE_14:
    unnested_all_vm_MSE_14.extend(mse_list)

# obtain quantile values
vm_min_14 = np.min(unnested_all_vm_MSE_14)
vm_q1_14 = np.quantile(unnested_all_vm_MSE_14, 0.25)
vm_median_14 = np.quantile(unnested_all_vm_MSE_14, 0.5)
vm_q3_14 = np.quantile(unnested_all_vm_MSE_14, 0.75)
vm_max_14 = np.max(unnested_all_vm_MSE_14)


### PG (CASE 30) ######################################################################
all_pg_MSE_30 = np.load("../all_pg_MSE_array_30_Kplus1.npy")

# verifying mean values match those in the table
pg_MSE_mean_list_30 = []
for mse_list in all_pg_MSE_30:
    pg_MSE_mean_list_30.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_pg_MSE_30 = []
for mse_list in all_pg_MSE_30:
    unnested_all_pg_MSE_30.extend(mse_list)

# obtain quantile values
pg_min_30 = np.min(unnested_all_pg_MSE_30)
pg_q1_30 = np.quantile(unnested_all_pg_MSE_30, 0.25)
pg_median_30 = np.quantile(unnested_all_pg_MSE_30, 0.5)
pg_q3_30 = np.quantile(unnested_all_pg_MSE_30, 0.75)
pg_max_30 = np.max(unnested_all_pg_MSE_30)


### QG (CASE 30) ######################################################################
all_qg_MSE_30 = np.load("../all_qg_MSE_array_30_Kplus1.npy")

# verifying mean values match those in the table
qg_MSE_mean_list_30 = []
for mse_list in all_qg_MSE_30:
    qg_MSE_mean_list_30.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_qg_MSE_30 = []
for mse_list in all_qg_MSE_30:
    unnested_all_qg_MSE_30.extend(mse_list)

# obtain quantile values
qg_min_30 = np.min(unnested_all_qg_MSE_30)
qg_q1_30 = np.quantile(unnested_all_qg_MSE_30, 0.25)
qg_median_30 = np.quantile(unnested_all_qg_MSE_30, 0.5)
qg_q3_30 = np.quantile(unnested_all_qg_MSE_30, 0.75)
qg_max_30 = np.max(unnested_all_qg_MSE_30)


### VA (CASE 30) ######################################################################
all_va_MSE_30 = np.load("../all_va_MSE_array_30_Kplus1.npy")

# verifying mean values match those in the table
va_MSE_mean_list_30 = []
for mse_list in all_va_MSE_30:
    va_MSE_mean_list_30.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_va_MSE_30 = []
for mse_list in all_va_MSE_30:
    unnested_all_va_MSE_30.extend(mse_list)

# obtain quantile values
va_min_30 = np.min(unnested_all_va_MSE_30)
va_q1_30 = np.quantile(unnested_all_va_MSE_30, 0.25)
va_median_30 = np.quantile(unnested_all_va_MSE_30, 0.5)
va_q3_30 = np.quantile(unnested_all_va_MSE_30, 0.75)
va_max_30 = np.max(unnested_all_va_MSE_30)


### VM (CASE 30) ######################################################################
all_vm_MSE_30 = np.load("../all_vm_MSE_array_30_Kplus1.npy")

# verifying mean values match those in the table
vm_MSE_mean_list_30 = []
for mse_list in all_vm_MSE_30:
    vm_MSE_mean_list_30.append(np.mean(mse_list))

# convert nested list into single list
unnested_all_vm_MSE_30 = []
for mse_list in all_vm_MSE_30:
    unnested_all_vm_MSE_30.extend(mse_list)

# obtain quantile values
vm_min_30 = np.min(unnested_all_vm_MSE_30)
vm_q1_30 = np.quantile(unnested_all_vm_MSE_30, 0.25)
vm_median_30 = np.quantile(unnested_all_vm_MSE_30, 0.5)
vm_q3_30 = np.quantile(unnested_all_vm_MSE_30, 0.75)
vm_max_30 = np.max(unnested_all_vm_MSE_30)

data = {}
data["pg"] = [
    dict(
        med=pg_median_14,
        q1=pg_q1_14,
        q3=pg_q3_14,
        whislo=pg_min_14,
        whishi=pg_max_14,
        fliers=[],
        label="pg (case 14)",
    ),
    dict(
        med=pg_median_30,
        q1=pg_q1_30,
        q3=pg_q3_30,
        whislo=pg_min_30,
        whishi=pg_max_30,
        fliers=[],
        label="pg (case 30)",
    ),
]
data["qg"] = [
    dict(
        med=qg_median_14,
        q1=qg_q1_14,
        q3=qg_q3_14,
        whislo=qg_min_14,
        whishi=qg_max_14,
        fliers=[],
        label="qg (case 14)",
    ),
    dict(
        med=qg_median_30,
        q1=qg_q1_30,
        q3=qg_q3_30,
        whislo=qg_min_30,
        whishi=qg_max_30,
        fliers=[],
        label="qg (case 30)",
    ),
]
data["va"] = [
    dict(
        med=va_median_14,
        q1=va_q1_14,
        q3=va_q3_14,
        whislo=va_min_14,
        whishi=va_max_14,
        fliers=[],
        label="va (case 14)",
    ),
    dict(
        med=va_median_30,
        q1=va_q1_30,
        q3=va_q3_30,
        whislo=va_min_30,
        whishi=va_max_30,
        fliers=[],
        label="va (case 30)",
    ),
]
data["vm"] = [
    dict(
        med=vm_median_14,
        q1=vm_q1_14,
        q3=vm_q3_14,
        whislo=vm_min_14,
        whishi=vm_max_14,
        fliers=[],
        label="vm (case 14)",
    ),
    dict(
        med=vm_median_30,
        q1=vm_q1_30,
        q3=vm_q3_30,
        whislo=vm_min_30,
        whishi=vm_max_30,
        fliers=[],
        label="vm (case 30)",
    ),
]


stats = []
for i in range(2):
    stats.append(data["pg"][i])
    stats.append(data["qg"][i])
    stats.append(data["va"][i])
    stats.append(data["vm"][i])

positions = np.array([0, 1, 2, 3, 4, 5, 6, 7])

fig, ax = plt.subplots()
bp = ax.bxp(stats, positions=positions, patch_artist=True)

colors = [
    "lightgreen",
    "lightyellow",
    "lightgreen",
    "lightyellow",
    "lightgreen",
    "lightyellow",
    "lightgreen",
    "lightyellow",
]

for i, box in enumerate(bp["boxes"]):
    box.set_facecolor(colors[i])

for median in bp["medians"]:
    median.set_color("black")

ax.set_xticks([0.5, 2.5, 4.5, 6.5])
ax.set_xticklabels(["pg", "qg", "va", "vm"])

for pos in [1.5, 3.5, 5.5]:
    ax.axvline(x=pos, color="black", linestyle="--", linewidth=1)

plt.yscale("log")

ax.set_xlabel("ACOPF Variables")
ax.set_ylabel("MSE")

plt.grid(True)

legend_labels = ["Case_14", "Case_30"]
legend_colors = ["lightgreen", "lightyellow"]
handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=label,
        markerfacecolor=color,
        markersize=10,
    )
    for label, color in zip(legend_labels, legend_colors)
]

ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.12, 1))

plt.title("Mean Squared Error (MSE): Adding a Line")
plt.savefig(f"mse_line.png")
plt.show()
