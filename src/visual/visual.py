# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# import numpy as np
# import os

# # ---------- 参数配置 ----------
# checkpoint_path = "your_model.ckpt"  # ← 修改为你的模型检查点路径
# routing_key = "encoder.layers.0.blocks.0.attn.adapter_instance.routing_weights"  # ← 修改为你的 routing 权重 key
# max_tsne_points = 3000  # 控制 t-SNE 可视化样本数，防止太大
# sample_id_for_heatmap = 0  # 热图中选择可视化的样本
# output_dir = "routing_visuals"
# os.makedirs(output_dir, exist_ok=True)


# # ---------- 加载路由权重 ----------
# ckpt = torch.load(checkpoint_path, map_location='cpu')
# if 'state_dict' in ckpt:
#     ckpt = ckpt['state_dict']

# # 获取 routing_weights（注意根据你模型结构修改 key）
# routing_weights = ckpt[routing_key]  # shape: [B, S, E]
# routing_weights = routing_weights.cpu()
# print(f"Routing Weights Shape: {routing_weights.shape}")

# B, S, E = routing_weights.shape


# # ---------- 1. T-SNE 可视化 ----------
# def plot_tsne(weights, save_path):
#     data = weights.view(-1, E).numpy()  # [B*S, E]

#     if data.shape[0] > max_tsne_points:
#         idx = np.random.choice(data.shape[0], size=max_tsne_points, replace=False)
#         data = data[idx]

#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     tsne_result = tsne.fit_transform(data)

#     plt.figure(figsize=(6, 6))
#     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10, alpha=0.6)
#     plt.title("T-SNE of Routing Weights")
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()


# # ---------- 2. Heatmap 可视化 ----------
# def plot_heatmap(weights, sample_id, save_path):
#     sample = weights[sample_id]  # shape [S, E]
#     plt.figure(figsize=(12, 5))
#     sns.heatmap(sample.numpy(), cmap="viridis", cbar=True)
#     plt.title(f"Routing Heatmap for Sample {sample_id}")
#     plt.xlabel("Expert Index")
#     plt.ylabel("Time Step")
#     plt.savefig(save_path)
#     plt.close()


# # ---------- 3. 每个 expert 平均使用频率 ----------
# def plot_expert_usage(weights, save_path):
#     mean_weights = weights.mean(dim=(0, 1)).numpy()  # shape: [E]
#     plt.figure(figsize=(6, 4))
#     plt.bar(np.arange(E), mean_weights)
#     plt.title("Average Routing Weight per Expert")
#     plt.xlabel("Expert Index")
#     plt.ylabel("Mean Weight")
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()


# # ---------- 可视化调用 ----------
# plot_tsne(routing_weights, os.path.join(output_dir, "tsne.png"))
# plot_heatmap(routing_weights, sample_id_for_heatmap, os.path.join(output_dir, f"heatmap_sample{sample_id_for_heatmap}.png"))
# plot_expert_usage(routing_weights, os.path.join(output_dir, "expert_usage.png"))

# print("✅ 可视化完成，结果保存在：", output_dir)
