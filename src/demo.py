import torch
import torch_dct as dct
 
# 创建一个随机 2-D 张量
x = torch.randn(100, 100)
print('x is ', x)

# 进行 2-D DCT-II 变换
X = dct.dct_2d(x)
print('X is ', X)
# 进行 2-D 逆 DCT-III 变换
y = dct.idct_2d(X)
print('y is ', y)

# 计算误差
error = (torch.abs(x - y)).sum()
print(f"最大误差: {torch.abs(x - y).max().item()}")
print(f"平均误差: {torch.abs(x - y).mean().item()}")
print(f"总误差: {error.item()}")
 
# 验证结果是否一致（使用更宽松的容差）
assert error < 1e-6  # x == y 在数值容差范围内