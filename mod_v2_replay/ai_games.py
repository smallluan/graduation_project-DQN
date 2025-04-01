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

# 预定义所有可能的动作列表
ALL_ACTIONS = [
    'pass',
    *[str(i) for i in range(1, 10)],  # 单张 1 - 9
    *[f"{i}{i}" for i in range(1, 10)],  # 对子 11 - 99
    *[f"{i}{i}{i}{j}" for i in range(1, 10) for j in range(1, 10) if j != i]  # 三带一 1112 等
]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}
OUTPUT_SIZE = len(ALL_ACTIONS)  # 模型输出层大小与动作数量一致


# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 定义纸牌游戏环境
class CardGameEnv:
    def __init__(self):
        self.reset()
        self.just_got_turn = True  # 标记是否刚获得出牌权
        self.last_played_pair = False  # 标记上一轮是否出的对子
        self.last_played_triple = False  # 标记上一轮是否出的三带一
        self.last_action_was_pass = False  # 标记上一轮是否是 pass

    def reset(self, player_hand=None, ai_hand=None, starting_player='player'):
        # 初始化牌堆
        if player_hand is None or ai_hand is None:
            self.deck = []
            for num in range(1, 10):
                self.deck.extend([num] * 4)
            random.shuffle(self.deck)
            self.player_hand = sorted(self.deck[:10])
            self.ai_hand = sorted(self.deck[10:20])
        else:
            # 使用传入的手牌
            self.player_hand = sorted(player_hand)
            self.ai_hand = sorted(ai_hand)

        # 初始化状态参数
        self.last_played = 0
        self.last_played_pair = False
        self.last_played_triple = False
        self.current_player = starting_player
        self.just_got_turn = True
        self.last_action_was_pass = False
        return self._get_state()

    def _get_state(self):
        # 增加对手可能的出牌历史（需维护历史记录）
        player_counts = [self.player_hand.count(i) for i in range(1, 10)]
        ai_counts = [self.ai_hand.count(i) for i in range(1, 10)]

        # 新增基于玩家手牌数量的推断
        player_max_card = max([i for i, cnt in enumerate(player_counts, 1) if cnt > 0], default=0) / 9
        opponent_pressure = sum(
            1 for i, cnt in enumerate(player_counts, 1) if cnt > 0 and i > self.last_played) / 10

        # 添加归一化处理
        state = np.concatenate([
            ai_counts, player_counts,
            [len(self.player_hand) / 10, len(self.ai_hand) / 10],  # 归一化手牌数量
            [self.last_played / 9, int(self.last_played_pair), int(self.last_played_triple)],  # 归一化last_played
            [player_max_card],  # 玩家可能的最大牌
            [opponent_pressure]  # 玩家可压制当前出牌的牌比例
        ])
        return state.astype(np.float32)

    def _has_pair(self, hand):
        return len(hand) >= 2 and hand[0] == hand[1]

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
        for triple in triples:
            remaining_hand = [card for card in hand if card != triple]
            for single in remaining_hand:
                triple_with_single.append(f"{triple}{triple}{triple}{single}")
        return triple_with_single

    def step(self, action):
        done = False
        if self.current_player == 'player':
            prev_player_hand = self.player_hand.copy()
            if self.just_got_turn and action == 'pass':
                reward = -100
            else:
                valid_actions = self._get_valid_actions(self.player_hand)
                if action == 'pass':
                    if self.just_got_turn or self.last_action_was_pass:
                        reward = -100
                    else:
                        valid_non_pass_actions = [a for a in valid_actions if a != 'pass']
                        if valid_non_pass_actions:
                            reward = -5
                        else:
                            reward = 0
                        # 更新状态
                        self.last_played = 0
                        self.last_played_pair = False
                        self.last_played_triple = False
                        self.current_player = 'ai'
                        self.just_got_turn = True  # AI 刚获得出牌权
                        self.last_action_was_pass = True
                else:
                    try:
                        if len(str(action)) == 2 and str(action)[0] == str(action)[1]:
                            # 出对子
                            card = int(str(action)[0])
                            if self.player_hand.count(card) < 2:
                                reward = -100
                            elif self.last_played_triple:
                                reward = -100
                            elif self.last_played_pair and card <= self.last_played:
                                reward = -100
                            elif self.last_played > 0 and card <= self.last_played:
                                reward = -100
                            else:
                                state = self._get_state()
                                player_max_card = state[-2]  # 从状态中获取玩家可能的最大牌
                                opponent_pressure = state[-1]  # 玩家可压制牌比例

                                # 动态调整奖励
                                hand_size_ratio = len(self.player_hand) / 10
                                base_reward = 20 * (1 - hand_size_ratio)
                                pressure_adjustment = 20 * (1 - opponent_pressure)
                                reward = base_reward + pressure_adjustment

                                self.player_hand.remove(card)
                                self.player_hand.remove(card)
                                self.last_played = card
                                self.last_played_pair = True
                                self.last_played_triple = False
                                self.last_action_was_pass = False

                                done = len(self.player_hand) == 0
                                if done and len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1]:
                                    reward += 20
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(
                                        self.player_hand) == 1:
                                    reward -= 3
                                if done:
                                    reward = -100
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                        elif len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
                            # 出三带一
                            triple_card = int(str(action)[0])
                            single_card = int(str(action)[3])
                            if self.player_hand.count(triple_card) < 3 or single_card not in self.player_hand:
                                reward = -100
                            elif self.last_played_triple and triple_card <= self.last_played:
                                reward = -100
                            elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                                reward = -100
                            else:
                                state = self._get_state()
                                triple_value = int(action[0])
                                opponent_pressure = state[-1]  # 玩家可压制牌比例

                                # 压制对手奖励
                                if triple_value > (state[-2] * 9):  # 玩家可能的最大牌
                                    pressure_bonus = 20
                                else:
                                    pressure_bonus = 0

                                # 动态调整
                                hand_size_ratio = len(self.player_hand) / 10
                                reward = 30 * (1 - hand_size_ratio) + pressure_bonus

                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(single_card)
                                self.last_played = triple_card
                                self.last_played_pair = False
                                self.last_played_triple = True
                                self.last_action_was_pass = False

                                done = len(self.player_hand) == 0
                                if done:
                                    reward = -100
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                        else:
                            # 出单牌
                            card = int(action)
                            if card not in self.player_hand:
                                reward = -100
                            elif self.last_played_triple:
                                reward = -100
                            elif self.last_played_pair:
                                reward = -100
                            elif card <= self.last_played:
                                reward = -100
                            else:
                                self.player_hand.remove(card)
                                self.last_played = card
                                self.last_played_pair = False
                                self.last_played_triple = False
                                self.last_action_was_pass = False
                                reward = 10
                                done = len(self.player_hand) == 0
                                if done and len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1]:
                                    reward += 20
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(
                                        self.player_hand) == 1:
                                    reward -= 3
                                if done:
                                    reward = -100
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                    except ValueError as e:
                        reward = -100
                    except Exception as e:
                        reward = -100
        else:
            prev_ai_hand = self.ai_hand.copy()
            valid_actions = self._get_valid_actions(self.ai_hand)
            if action == 'pass':
                if self.just_got_turn or self.last_action_was_pass:
                    reward = -100
                else:
                    valid_non_pass_actions = [a for a in valid_actions if a != 'pass']
                    if valid_non_pass_actions:
                        reward = -5
                    else:
                        reward = 0
                    # 更新状态
                    self.last_played = 0
                    self.last_played_pair = False
                    self.last_played_triple = False
                    self.current_player = 'player'
                    self.just_got_turn = True  # 玩家刚获得出牌权
                    self.last_action_was_pass = True
            else:
                if len(action) == 2 and action[0] == action[1]:
                    # 出对子
                    card = int(action[0])
                    if self.ai_hand.count(card) < 2:
                        reward = -100
                    elif self.last_played_triple and card <= self.last_played:
                        reward = -100
                    elif self.last_played_pair and card <= self.last_played:
                        reward = -100
                    elif self.last_played > 0 and card <= self.last_played:
                        reward = -100
                    else:
                        state = self._get_state()
                        player_max_card = state[-2]  # 从状态中获取玩家可能的最大牌
                        opponent_pressure = state[-1]  # 玩家可压制牌比例

                        # 动态调整奖励
                        hand_size_ratio = len(self.ai_hand) / 10
                        base_reward = 10 * (1 - hand_size_ratio)
                        pressure_adjustment = 20 * (1 - opponent_pressure)
                        reward = base_reward + pressure_adjustment

                        self.ai_hand.remove(card)
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = True
                        self.last_played_triple = False
                        self.last_action_was_pass = False

                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += 20
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward -= 1
                        if done:
                            reward = 100
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

                elif len(action) == 4 and action[:3] == action[0] * 3:
                    # 出三带一
                    triple_card = int(action[0])
                    single_card = int(action[3])
                    if self.ai_hand.count(triple_card) < 3 or single_card not in self.ai_hand:
                        reward = -100
                    elif self.last_played_triple and triple_card <= self.last_played:
                        reward = -100
                    elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                        reward = -100
                    else:
                        state = self._get_state()
                        triple_value = int(action[0])
                        opponent_pressure = state[-1]  # 玩家可压制牌比例

                        # 压制对手奖励
                        if triple_value > (state[-2] * 9):  # 玩家可能的最大牌
                            pressure_bonus = 20
                        else:
                            pressure_bonus = 0

                        # 动态调整
                        hand_size_ratio = len(self.ai_hand) / 10
                        reward = 15 * (1 - hand_size_ratio) + pressure_bonus

                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(single_card)
                        self.last_played = triple_card
                        self.last_played_pair = False
                        self.last_played_triple = True
                        self.last_action_was_pass = False

                        done = len(self.ai_hand) == 0
                        if done:
                            reward = 100
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

                else:
                    # 出单牌
                    card = int(action)
                    if card not in self.ai_hand:
                        reward = -100
                    elif self.last_played_triple:
                        reward = -100
                    elif self.last_played_pair:
                        reward = -100
                    elif card <= self.last_played:
                        reward = -100
                    else:
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = False
                        self.last_played_triple = False
                        self.last_action_was_pass = False
                        reward = 5
                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += 20
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward -= 1
                        if done:
                            reward = 100
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

        # 新增动态惩罚机制
        # 手牌数量惩罚（玩家视角）
        if self.current_player == 'player':
            hand_size_penalty = 0.5 * (len(self.player_hand) / 10)
            reward -= hand_size_penalty
        else:
            hand_size_penalty = 0.5 * (len(self.ai_hand) / 10)
            reward -= hand_size_penalty

        # 过早出牌惩罚（基于游戏阶段）
        if self.current_player == 'player':
            game_stage = (10 - len(self.player_hand)) / 10
        else:
            game_stage = (10 - len(self.ai_hand)) / 10

        action_type = 'pair' if len(action) == 2 and action[0] == action[1] else 'triple' if len(
            action) == 4 and action[:3] == action[0] * 3 else None
        if action_type == 'pair' and game_stage < 0.6:
            reward *= (1 - game_stage * 0.8)

        # 根据牌数调整奖惩强度
        if self.current_player == 'player':
            opponent_cards = len(self.ai_hand)
            own_cards = len(self.player_hand)
            if opponent_cards < 5:
                reward -= opponent_cards * 5
            if own_cards < 5:
                reward += (5 - own_cards) * 5
        else:
            opponent_cards = len(self.player_hand)
            own_cards = len(self.ai_hand)
            if opponent_cards < 5:
                reward -= opponent_cards * 5
            if own_cards < 5:
                reward += (5 - own_cards) * 5

        if not done:
            self.just_got_turn = False

        next_state = self._get_state()
        return next_state, reward, done

    def _get_valid_actions(self, hand):
        valid_actions = []

        if self.just_got_turn or self.last_action_was_pass:
            # 抢占了出牌权，必须出牌，不能 pass
            # 生成所有合法的出牌动作
            valid_single = [str(c) for c in hand]
            valid_triple = []
            triples = self._find_triples(hand)
            for triple in triples:
                remaining_hand = [card for card in hand if card != triple]
                for single in remaining_hand:
                    valid_triple.append(f"{triple}{triple}{triple}{single}")
            valid_pair = [f"{c}{c}" for c in self._find_pairs(hand)]
            valid_actions = valid_single + valid_triple + valid_pair
        else:
            # 未抢占出牌权，可以出牌或 pass
            if self.last_played_triple:
                # 必须出更大的三带一
                valid_actions = self._find_triple_with_single(hand)
                valid_actions = [act for act in valid_actions if int(act[:3][0]) > self.last_played]
            elif self.last_played_pair:
                # 必须出更大的对子
                pairs = self._find_pairs(hand)
                valid_actions = [f"{c}{c}" for c in pairs if c > self.last_played]
            else:
                if self.last_played > 0:
                    # 必须出更大的单牌
                    valid_single = [str(c) for c in hand if c > self.last_played]
                    valid_actions = valid_single
                else:
                    # 可以出任意合法的牌
                    valid_single = [str(c) for c in hand]
                    valid_triple = []
                    triples = self._find_triples(hand)
                    for triple in triples:
                        remaining_hand = [card for card in hand if card != triple]
                        for single in remaining_hand:
                            valid_triple.append(f"{triple}{triple}{triple}{single}")
                    valid_pair = [f"{c}{c}" for c in self._find_pairs(hand)]
                    valid_actions = valid_single + valid_triple + valid_pair

            # 未抢占出牌权，允许 pass
            valid_actions.append('pass')

        # 如果没有合法动作，强制 pass
        if not valid_actions:
            valid_actions = ['pass']

        return valid_actions


# 加载模型
def load_model(input_size, model_path):
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 评估模型
def evaluate_model(model, env, num_episodes=l):
    total_wins = 0
    total_rewards = []
    win_rates = []
    avg_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if env.current_player == 'player':
                valid_actions = env._get_valid_actions(env.player_hand)
                if not valid_actions:
                    action = 'pass'
                else:
                    action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = model(state)
                valid_actions = env._get_valid_actions(env.ai_hand)
                valid_action_indices = {action: ACTION_TO_INDEX[action] for action in valid_actions if
                                        action in ACTION_TO_INDEX}
                if not valid_actions:
                    action = 'pass'
                else:
                    valid_q_values = [q_values[0][valid_action_indices[action]] for action in valid_actions if
                                      action in valid_action_indices]
                    action = valid_actions[torch.argmax(torch.tensor(valid_q_values)).item()]

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward
            state = next_state

        if len(env.ai_hand) == 0:
            total_wins += 1
            total_rewards.append(WIN_REWARD)
        else:
            total_rewards.append(LOSS_REWARD)

        win_rate = total_wins / (episode + 1)
        win_rates.append(win_rate)
        avg_reward = sum(total_rewards) / (episode + 1)
        avg_rewards.append(avg_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Win Rate: {win_rate:.4f}, Average Reward: {avg_reward:.4f}")

    # 绘制胜率和平均奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(win_rates)
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return win_rates, avg_rewards


if __name__ == "__main__":
    env = CardGameEnv()
    input_size = len(env._get_state())
    model_path = "models/model10000.pth"  # 请根据实际情况修改模型路径
    loaded_model = load_model(input_size, model_path)

    win_rates, avg_rewards = evaluate_model(loaded_model, env)

    # 最终胜率和平均奖励
    final_win_rate = win_rates[-1]
    final_avg_reward = avg_rewards[-1]
    print(f"Final Win Rate: {final_win_rate:.4f}")
    print(f"Final Average Reward: {final_avg_reward:.4f}")

    # 胜率和平均奖励的置信区间
    win_rate_ci = stats.t.interval(0.95, len(win_rates) - 1, loc=np.mean(win_rates), scale=stats.sem(win_rates))
    avg_reward_ci = stats.t.interval(0.95, len(avg_rewards) - 1, loc=np.mean(avg_rewards), scale=stats.sem(avg_rewards))
    print(f"Win Rate 95% Confidence Interval: {win_rate_ci}")
    print(f"Average Reward 95% Confidence Interval: {avg_reward_ci}")
