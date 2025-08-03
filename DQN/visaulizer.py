import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件，只加载需要的列
df = pd.read_csv('./logs/Breakout_log.csv', usecols=['Step', 'Reward', 'Loss'])

# 可选：按 Step 排序（如果不保证有序）
df = df.sort_values(by='Step')

# 绘制 Reward
plt.figure(figsize=(10, 4))
plt.plot(df['Step'], df['Reward'], label='Reward', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Step vs Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制 Loss
plt.figure(figsize=(10, 4))
plt.plot(df['Step'], df['Loss'], label='Loss', color='orange', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Step vs Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
