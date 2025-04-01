import os
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 全局参数
l = 1000  # 对局次数
# 定义奖励值
WIN_REWARD = 1
LOSS_REWARD = -1

# ---------------------- 预定义所有可能的动作列表 ----------------------
# 必须与训练模型时的动作列表完全一致！
ALL_ACTIONS = [
    'pass',
    *[str(i) for i in range(1, 10)],  # 单张 1-9
    *[f"{i}{i}" for i in range(1, 10)],  # 对子 11-99
    *[f"{i}{i}{i}{j}" for i in range(1, 10) for j in range(1, 10) if j != i]  # 三带一 1112等
]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}
OUTPUT_SIZE = len(ALL_ACTIONS)  # 模型输出层大小与动作数量一致

# 模型路径数组
model_paths = [
    "models/model10.pth",
    "models/model50.pth",
    "models/model100.pth",
    "models/model500.pth",
    "models/model1000.pth",
    "models/model5000.pth",
    "models/model10000.pth"
]

# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, OUTPUT_SIZE)  # 修改输出层大小

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 纸牌游戏环境
class CardGameEnv:
    def __init__(self, player1_hand, player2_hand):
        self.player1_hand = player1_hand.copy()
        self.player2_hand = player2_hand.copy()
        self.last_played = 0
        self.last_played_pair = False
        self.last_played_triple = False
        self.current_player = 'player1'
        self.just_got_turn = True

    def reset(self, player1_hand, player2_hand):
        self.player1_hand = player1_hand.copy()
        self.player2_hand = player2_hand.copy()
        self.last_played = 0
        self.last_played_pair = False
        self.last_played_triple = False
        self.current_player = 'player1'
        self.just_got_turn = True
        return self._get_state()

    def _get_state(self):
        current_hand = self.player1_hand if self.current_player == 'player1' else self.player2_hand
        opponent_hand = self.player2_hand if self.current_player == 'player1' else self.player1_hand
        current_hand = current_hand.copy()
        current_hand += [0] * (10 - len(current_hand))
        current_hand = current_hand[:10]
        state = current_hand + [len(opponent_hand), self.last_played, int(self.last_played_pair),
                                int(self.last_played_triple)]
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def _has_pair(self, hand):
        for i in range(len(hand) - 1):
            if hand[i] == hand[i + 1]:
                return True
        return False

    def _find_pairs(self, hand):
        pairs = []
        for i in range(len(hand) - 1):
            if hand[i] == hand[i + 1]:
                pairs.append(hand[i])
        return pairs

    def _has_triple(self, hand):
        for i in range(len(hand) - 2):
            if hand[i] == hand[i + 1] == hand[i + 2]:
                return True
        return False

    def _find_triples(self, hand):
        triples = []
        for i in range(len(hand) - 2):
            if hand[i] == hand[i + 1] == hand[i + 2]:
                triples.append(hand[i])
        return triples

    def _find_triple_with_single(self, hand):
        triples = self._find_triples(hand)
        triple_with_single = []
        for t in triples:
            remaining = [c for c in hand if c != t]
            for s in remaining:
                triple_with_single.append(f"{t}{t}{t}{s}")
        return triple_with_single

    def step(self, action):
        done = False
        current_hand = self.player1_hand if self.current_player == 'player1' else self.player2_hand
        valid_actions = self.get_valid_actions(current_hand)

        if self.just_got_turn and action == 'pass':
            return self._get_state(), 0, done
        elif action == 'pass':
            if valid_actions == ['pass']:
                self.last_played = 0
                self.last_played_pair = False
                self.last_played_triple = False
                next_player = 'player2' if self.current_player == 'player1' else 'player1'
                self.current_player = next_player
                self.just_got_turn = True
            else:
                return self._get_state(), 0, done
        else:
            if action not in valid_actions:
                return self._get_state(), 0, done

            try:
                if len(str(action)) == 2 and str(action)[0] == str(action)[1]:
                    card = int(str(action)[0])
                    current_hand.remove(card)
                    current_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = True
                    self.last_played_triple = False
                    done = len(current_hand) == 0
                    next_player = 'player2' if self.current_player == 'player1' else 'player1'
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
                elif len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
                    triple_card = int(str(action)[0])
                    single_card = int(str(action)[3])
                    current_hand.remove(triple_card)
                    current_hand.remove(triple_card)
                    current_hand.remove(triple_card)
                    current_hand.remove(single_card)
                    self.last_played = triple_card
                    self.last_played_pair = False
                    self.last_played_triple = True
                    done = len(current_hand) == 0
                    next_player = 'player2' if self.current_player == 'player1' else 'player1'
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
                else:
                    card = int(action)
                    current_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = False
                    self.last_played_triple = False
                    done = len(current_hand) == 0
                    next_player = 'player2' if self.current_player == 'player1' else 'player1'
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
            except:
                return self._get_state(), 0, done

        if not done:
            self.just_got_turn = False

        next_state = self._get_state()
        if done:
            reward = WIN_REWARD if len(self.player1_hand) == 0 else LOSS_REWARD
        else:
            reward = 0
        return next_state, reward, done

    def get_valid_actions(self, hand):
        valid_actions = []
        if self.last_played == 0:
            valid_single = [str(c) for c in hand]
            valid_actions = valid_single
            pairs = self._find_pairs(hand)
            valid_actions.extend([str(c) * 2 for c in pairs])
            triple_with_single = self._find_triple_with_single(hand)
            valid_actions.extend(triple_with_single)
        elif self.last_played_triple:
            valid_actions = self._find_triple_with_single(hand)
            valid_actions = [act for act in valid_actions if int(act[:3][0]) > self.last_played]
        elif self.last_played_pair:
            pairs = self._find_pairs(hand)
            valid_actions = [str(c) * 2 for c in pairs if c > self.last_played]
        else:
            valid_single = [str(c) for c in hand if c > self.last_played]
            valid_actions = valid_single

        if self.just_got_turn:
            valid_actions = [a for a in valid_actions if a != 'pass']

        if not valid_actions:
            valid_actions = ['pass']

        return valid_actions


# 加载模型
def load_model(input_size, model_path):
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 固定策略函数
def fixed_strategy(env, current_hand):
    valid_actions = env.get_valid_actions(current_hand)
    if 'pass' in valid_actions:
        return 'pass'
    for action in valid_actions:
        if len(str(action)) == 2 and str(action)[0] == str(action)[1]:
            return action
    for action in valid_actions:
        if len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
            return action
    return valid_actions[0]


# 进行对局
def play_games(env, model1):
    m1_wins = 0
    rewards = []
    win_rates = []
    input_size = len(env._get_state()[0])

    for i in range(l):
        if i < len(hand_data):
            player1_hand, player2_hand = hand_data[i]
        else:
            original_idx = i - len(hand_data)
            player1_hand, player2_hand = hand_data[original_idx][1], hand_data[original_idx][0]

        state = env.reset(player1_hand, player2_hand)
        done = False
        total_reward = 0

        while not done:
            if env.current_player == 'player1':
                q_values = model1(state)
                current_hand = env.player1_hand
                valid_actions = env.get_valid_actions(current_hand)
                valid_indices = [ACTION_TO_INDEX[action] for action in valid_actions if action in ACTION_TO_INDEX]
                if not valid_indices:
                    action = 'pass'
                else:
                    valid_q = q_values[0][valid_indices]
                    best_idx = torch.argmax(valid_q).item()
                    action = valid_actions[best_idx]
            else:
                current_hand = env.player2_hand
                action = fixed_strategy(env, current_hand)

            next_state, reward, done = env.step(action)
            state = next_state
            if env.current_player == 'player1':
                total_reward += reward

        if len(env.player1_hand) == 0:
            m1_wins += 1

        win_rate = (m1_wins / (i + 1)) * 100
        win_rates.append(win_rate)
        rewards.append(total_reward)

        print(f" 完成第 { i + 1 }轮")

    # 计算核心指标
    win_rate = (m1_wins / l) * 100
    average_reward = np.mean(rewards)
    # 学习曲线即胜率随对局次数的变化
    learning_curve = win_rates

    # 计算辅助指标
    reward_std = np.std(rewards)
    # 这里简单假设关键决策是出牌大于对手，由于没有明确标准，可根据实际情况修改
    key_decision_accuracy = calculate_key_decision_accuracy(env, model1)
    # 对抗性测试，这里简单用胜率表示，可根据需求扩展
    adversarial_test_result = win_rate

    # 统计验证
    confidence_interval = stats.t.interval(0.95, len(rewards) - 1, loc=average_reward, scale=stats.sem(rewards))
    # 这里假设与随机策略对比，p 值检验简单模拟，可根据实际情况修改
    p_value = stats.ttest_1samp(rewards, 0).pvalue

    print(f"胜率: {win_rate}%")
    print(f"平均奖励: {average_reward}")
    print(f"奖励标准差: {reward_std}")
    print(f"关键决策准确率: {key_decision_accuracy}%")
    print(f"对抗性测试结果: {adversarial_test_result}%")
    print(f"平均奖励 95% 置信区间: {confidence_interval}")
    print(f"p 值: {p_value}")

    return win_rate, average_reward, learning_curve


# 计算关键决策准确率
def calculate_key_decision_accuracy(env, model1):
    correct_decisions = 0
    total_decisions = 0
    for i in range(l):
        if i < len(hand_data):
            player1_hand, player2_hand = hand_data[i]
        else:
            original_idx = i - len(hand_data)
            player1_hand, player2_hand = hand_data[original_idx][1], hand_data[original_idx][0]

        state = env.reset(player1_hand, player2_hand)
        done = False

        while not done:
            if env.current_player == 'player1':
                q_values = model1(state)
                current_hand = env.player1_hand
                valid_actions = env.get_valid_actions(current_hand)
                valid_indices = [ACTION_TO_INDEX[action] for action in valid_actions if action in ACTION_TO_INDEX]
                if valid_indices:
                    valid_q = q_values[0][valid_indices]
                    best_idx = torch.argmax(valid_q).item()
                    action = valid_actions[best_idx]
                    total_decisions += 1
                    if env.last_played > 0 and action != 'pass':
                        if len(str(action)) == 1:
                            if int(action) > env.last_played:
                                correct_decisions += 1
                        elif len(str(action)) == 2:
                            if int(str(action)[0]) > env.last_played:
                                correct_decisions += 1
                        elif len(str(action)) == 4:
                            if int(str(action)[0]) > env.last_played:
                                correct_decisions += 1

            next_state, _, done = env.step(
                action if env.current_player == 'player1' else fixed_strategy(env, env.player2_hand))
            state = next_state

    if total_decisions == 0:
        return 0
    return (correct_decisions / total_decisions) * 100


# 生成测试数据
def setData(l):
    deck = [i for i in range(1, 10) for _ in range(4)]
    hand_data = []
    for _ in range(l // 2):
        random.shuffle(deck)
        hand_data.append((sorted(deck[:10]), sorted(deck[10:20])))
    with open('test_data/data.py', 'w') as f:
        f.write(f"hand_data = {hand_data}")


if __name__ == "__main__":
    # 初始化环境
    if not os.path.exists('test_data/data.py'):
        setData(l)
    from test_data.data import hand_data

    # 示例初始化
    env = CardGameEnv([], [])  # 手牌通过reset设置
    input_size = len(env._get_state()[0])

    all_learning_curves = []
    for model_path in model_paths:
        model1 = load_model(input_size, model_path)
        _, _, learning_curve = play_games(env, model1)
        all_learning_curves.append(learning_curve)

    # 绘制所有模型的胜率变化曲线
    plt.figure(figsize=(12, 6))
    for i, learning_curve in enumerate(all_learning_curves):
        plt.plot(range(1, l + 1), learning_curve, label=f'{model_paths[i][7: -4]}')
    plt.title('Each model wins')
    plt.xlabel('count')
    plt.ylabel('wins (%)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 70)
    plt.tight_layout()
    plt.show()