import numpy as np
import random
import DiscountP as DP

collector = {}
"""
实验开始部分，表示进行n次实验，其中参数c代表cost，K代表卖出收益，n表示状态数，p是转移概率
a代表alpha，折扣因子，通过随机生成
"""
for i in range(1):
    c, K, n = random.sample([random.randint(1, 10) for _ in range(1000)], 3)
    p, a = np.random.random(2)
    print(c, K, n, p, a)
    collector[str(i) + " " + "policy_iteration"] = DP.DiscountProblem(c, K, n, p, a).policy_iteration()
    collector[str(i) + " " + "value_iteration"] = DP.DiscountProblem(c, K, n, p, a).value_Iteration()
print("=====================")
for key in collector:
    print(key+" ",collector[key])