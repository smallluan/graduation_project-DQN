## 说明

**作者 —— 田硕**

### 一、 模型说明
```python
# 全局参数
num_episodes = 1000  # 训练轮数
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
learning_rate = 0.001  # 学习率

# 奖励相关参数
reward_invalid_action = -0.5  # 无效动作奖励
reward_pass_with_valid = -0.5  # 有有效动作时 pass 的奖励
reward_pass_without_valid = -0.1  # 无有效动作时 pass 的奖励
reward_single_card = 0.1  # 出单牌奖励
reward_pair_card = 0.5  # 出对子奖励
reward_triple_with_single = 0.7  # 出三带一奖励
reward_player_win = -1  # 玩家获胜奖励
reward_ai_win = 1  # AI 获胜奖励
```

### 二、 模型训练 —— main.py
1. 静态学习参数：学习参数不会随着局势的推进而变化
2. 静态奖励参数：奖励参数不会随着局势的推进而变化
   - 优点：开发方便
   - 缺点：错过最优解
   - 解决办法：设置动态奖励参数，如：AI还剩两张牌并且是对子，且本轮 AI 出牌，则AI直接出对子，+ 1分，AI将对子拆开出 - 1分
3. 
### 三、 模型对抗性测试 —— ai_games.py

### 四、 测试数据集 —— test_data

### 五、模型存储 —— models
