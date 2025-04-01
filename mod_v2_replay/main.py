import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

# 全局参数
num_episodes = 1000  # 训练轮数
initial_gamma = 0.99  # 初始折扣因子
gamma_decay = 0.999  # 折扣因子衰减率
initial_epsilon = 0.5  # 初始探索率
epsilon_decay = 0.995  # 探索率衰减率
epsilon_min = 0.2  # 最小探索率
learning_rate = 0.0001  # 学习率
batch_size = 64  # 经验回放批次大小
memory_capacity = 10000  # 经验回放容量
target_update_freq = 500  # 目标网络更新频率

# 奖励相关参数
reward_invalid_action = -2  # 无效动作奖励，惩罚加重
reward_pass_with_valid = -1  # 有有效动作时 pass 的奖励，惩罚加重
reward_pass_without_valid = -0.5  # 无有效动作时 pass 的奖励
reward_single_card = 1  # 出单牌奖励，适当提高
reward_pair_card = 3  # 出对子奖励，适当提高
reward_triple_with_single = 5  # 出三带一奖励，适当提高
reward_player_win = -10  # 玩家获胜奖励，惩罚加重
reward_ai_win = 10  # AI 获胜奖励，奖励加重
reward_pair_win_bonus = 5  # 对子获胜额外奖励，提高
reward_split_pair_penalty = -5  # 拆对子惩罚，加重
reward_split_triple_penalty = -5  # 拆三带一惩罚，加重
reward_pass_avoid_split = 3  # pass避免拆牌的奖励

# AI 相关奖励参数
reward_ai_single_card = 1
reward_ai_pair_card = 3
reward_ai_triple_with_single = 3
reward_ai_split_pair_penalty = -5
reward_ai_split_triple_penalty = -5
reward_ai_pass_avoid_split = 3
reward_ai_low_cards_bonus = 5
reward_opponent_low_cards_penalty = 3
reward_ai_single_penalty = 5  # 出的单子比可出的单子大，需要惩罚，避免过早出掉大牌
reward_ai_pair_penalty = 5  # 出的单对子比可出的对子大，需要惩罚，避免过早出掉大牌
reward_ai_triple_penalty = 5  # 出的三带一比可出的三带一大，需要惩罚，避免过早出掉大牌

# 新增参数
reward_hand_size_penalty = 0.5  # 手牌数量惩罚系数
reward_opponent_pressure_bonus = 5  # 压制对手时的额外奖励
reward_premature_play_penalty = 0.8  # 过早出牌惩罚系数

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


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


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
                reward = reward_invalid_action
            else:
                valid_actions = self._get_valid_actions(self.player_hand)
                if action == 'pass':
                    if self.just_got_turn or self.last_action_was_pass:
                        reward = reward_invalid_action
                    else:
                        valid_non_pass_actions = [a for a in valid_actions if a != 'pass']
                        if valid_non_pass_actions:
                            reward = reward_pass_with_valid
                            # 检查是否避免拆牌
                            has_pair = len(self._find_pairs(self.player_hand)) > 0
                            has_triple = len(self._find_triples(self.player_hand)) > 0
                            if has_pair or has_triple:
                                reward += reward_pass_avoid_split
                        else:
                            reward = reward_pass_without_valid
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
                                reward = reward_invalid_action
                            elif self.last_played_triple:
                                reward = reward_invalid_action
                            elif self.last_played_pair and card <= self.last_played:
                                reward = reward_invalid_action
                            elif self.last_played > 0 and card <= self.last_played:
                                reward = reward_invalid_action
                            else:
                                state = self._get_state()
                                player_max_card = state[-2]  # 从状态中获取玩家可能的最大牌
                                opponent_pressure = state[-1]  # 玩家可压制牌比例

                                # 动态调整奖励
                                hand_size_ratio = len(self.player_hand) / 10
                                base_reward = reward_pair_card * (1 - hand_size_ratio)
                                pressure_adjustment = reward_opponent_pressure_bonus * (1 - opponent_pressure)
                                reward = base_reward + pressure_adjustment

                                self.player_hand.remove(card)
                                self.player_hand.remove(card)
                                self.last_played = card
                                self.last_played_pair = True
                                self.last_played_triple = False
                                self.last_action_was_pass = False

                                done = len(self.player_hand) == 0
                                if done and len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1]:
                                    reward += reward_pair_win_bonus
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(
                                        self.player_hand) == 1:
                                    reward += reward_split_pair_penalty
                                if done:
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                                # 对子惩罚逻辑
                                if not self._has_triple(self.player_hand):
                                    valid_pairs = self._find_pairs(self.player_hand)
                                    valid_pairs = [p for p in valid_pairs if p > self.last_played]
                                    if valid_pairs:
                                        min_pair = min(valid_pairs)
                                        if card > min_pair:
                                            reward -= 0.5

                        elif len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
                            # 出三带一
                            triple_card = int(str(action)[0])
                            single_card = int(str(action)[3])
                            if self.player_hand.count(triple_card) < 3 or single_card not in self.player_hand:
                                reward = reward_invalid_action
                            elif self.last_played_triple and triple_card <= self.last_played:
                                reward = reward_invalid_action
                            elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                                reward = reward_invalid_action
                            else:
                                state = self._get_state()
                                triple_value = int(action[0])
                                opponent_pressure = state[-1]  # 玩家可压制牌比例

                                # 压制对手奖励
                                if triple_value > (state[-2] * 9):  # 玩家可能的最大牌
                                    pressure_bonus = reward_opponent_pressure_bonus
                                else:
                                    pressure_bonus = 0

                                # 动态调整
                                hand_size_ratio = len(self.player_hand) / 10
                                reward = reward_triple_with_single * (1 - hand_size_ratio) + pressure_bonus

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
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                                # 三代一奖惩逻辑
                                possible_triples = self._find_triples(self.player_hand)
                                if possible_triples:
                                    min_triple = min(possible_triples)
                                    if triple_card > min_triple:
                                        reward -= 0.5

                                remaining_hand = [c for c in self.player_hand if c != triple_card]
                                if remaining_hand:
                                    min_single = min(remaining_hand)
                                    if single_card > min_single:
                                        reward -= 0.5

                        else:
                            # 出单牌
                            card = int(action)
                            if card not in self.player_hand:
                                reward = reward_invalid_action
                            elif self.last_played_triple:
                                reward = reward_invalid_action
                            elif self.last_played_pair:
                                reward = reward_invalid_action
                            elif card <= self.last_played:
                                reward = reward_invalid_action
                            else:
                                self.player_hand.remove(card)
                                self.last_played = card
                                self.last_played_pair = False
                                self.last_played_triple = False
                                self.last_action_was_pass = False
                                reward = reward_single_card
                                done = len(self.player_hand) == 0
                                if done and len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1]:
                                    reward += reward_pair_win_bonus
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(
                                        self.player_hand) == 1:
                                    reward += reward_split_pair_penalty
                                if done:
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True

                                # 拆对子/三张惩罚逻辑
                                prev_count = prev_player_hand.count(card)
                                if prev_count >= 2:
                                    reward -= 5  # 拆对子或三张惩罚

                                # 单牌惩罚逻辑
                                if not self._has_triple(self.player_hand):
                                    valid_single = [c for c in self.player_hand if c > self.last_played]
                                    if valid_single:
                                        min_single = min(valid_single)
                                        if card > min_single:
                                            reward -= 0.5

                    except ValueError as e:
                        reward = reward_invalid_action
                    except Exception as e:
                        reward = reward_invalid_action
        else:
            prev_ai_hand = self.ai_hand.copy()
            valid_actions = self._get_valid_actions(self.ai_hand)
            if action == 'pass':
                if self.just_got_turn or self.last_action_was_pass:
                    reward = reward_invalid_action
                else:
                    valid_non_pass_actions = [a for a in valid_actions if a != 'pass']
                    if valid_non_pass_actions:
                        reward = reward_pass_with_valid
                        # 检查是否避免拆牌
                        has_pair = len(self._find_pairs(self.ai_hand)) > 0
                        has_triple = len(self._find_triples(self.ai_hand)) > 0
                        if has_pair or has_triple:
                            reward += reward_ai_pass_avoid_split
                    else:
                        reward = reward_pass_without_valid
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
                        reward = reward_invalid_action
                    elif self.last_played_triple and card <= self.last_played:
                        reward = reward_invalid_action
                    elif self.last_played_pair and card <= self.last_played:
                        reward = reward_invalid_action
                    elif self.last_played > 0 and card <= self.last_played:
                        reward = reward_invalid_action
                    else:
                        state = self._get_state()
                        player_max_card = state[-2]  # 从状态中获取玩家可能的最大牌
                        opponent_pressure = state[-1]  # 玩家可压制牌比例

                        # 动态调整奖励
                        hand_size_ratio = len(self.ai_hand) / 10
                        base_reward = reward_ai_pair_card * (1 - hand_size_ratio)
                        pressure_adjustment = reward_opponent_pressure_bonus * (1 - opponent_pressure)
                        reward = base_reward + pressure_adjustment

                        self.ai_hand.remove(card)
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = True
                        self.last_played_triple = False
                        self.last_action_was_pass = False

                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += reward_pair_win_bonus
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward += reward_ai_split_pair_penalty
                        if done:
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

                        # 对子惩罚逻辑
                        if not self._has_triple(self.ai_hand):
                            valid_pairs = self._find_pairs(self.ai_hand)
                            valid_pairs = [p for p in valid_pairs if p > self.last_played]
                            if valid_pairs:
                                min_pair = min(valid_pairs)
                                if card > min_pair:
                                    reward -= reward_ai_pair_penalty

                elif len(action) == 4 and action[:3] == action[0] * 3:
                    # 出三带一
                    triple_card = int(action[0])
                    single_card = int(action[3])
                    if self.ai_hand.count(triple_card) < 3 or single_card not in self.ai_hand:
                        reward = reward_invalid_action
                    elif self.last_played_triple and triple_card <= self.last_played:
                        reward = reward_invalid_action
                    elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                        reward = reward_invalid_action
                    else:
                        state = self._get_state()
                        triple_value = int(action[0])
                        opponent_pressure = state[-1]  # 玩家可压制牌比例

                        # 压制对手奖励
                        if triple_value > (state[-2] * 9):  # 玩家可能的最大牌
                            pressure_bonus = reward_opponent_pressure_bonus
                        else:
                            pressure_bonus = 0

                        # 动态调整
                        hand_size_ratio = len(self.ai_hand) / 10
                        reward = reward_ai_triple_with_single * (1 - hand_size_ratio) + pressure_bonus

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
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

                        # 三代一奖惩逻辑
                        possible_triples = self._find_triples(self.ai_hand)
                        if possible_triples:
                            min_triple = min(possible_triples)
                            if triple_card > min_triple:
                                reward -= reward_ai_triple_penalty

                        remaining_hand = [c for c in self.ai_hand if c != triple_card]
                        if remaining_hand:
                            min_single = min(remaining_hand)
                            if single_card > min_single:
                                reward -= reward_ai_single_penalty

                else:
                    # 出单牌
                    card = int(action)
                    if card not in self.ai_hand:
                        reward = reward_invalid_action
                    elif self.last_played_triple:
                        reward = reward_invalid_action
                    elif self.last_played_pair:
                        reward = reward_invalid_action
                    elif card <= self.last_played:
                        reward = reward_invalid_action
                    else:
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = False
                        self.last_played_triple = False
                        self.last_action_was_pass = False
                        reward = reward_ai_single_card
                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += reward_pair_win_bonus
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward += reward_ai_split_pair_penalty
                        if done:
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

                        # 拆对子/三张惩罚逻辑
                        prev_count = prev_ai_hand.count(card)
                        if prev_count >= 2:
                            reward += reward_ai_split_pair_penalty

                        # 单牌惩罚逻辑
                        if not self._has_triple(self.ai_hand):
                            valid_single = [c for c in self.ai_hand if c > self.last_played]
                            if valid_single:
                                min_single = min(valid_single)
                                if card > min_single:
                                    reward -= reward_ai_single_penalty

        # 新增动态惩罚机制
        # 手牌数量惩罚（玩家视角）
        if self.current_player == 'player':
            hand_size_penalty = reward_hand_size_penalty * (len(self.player_hand) / 10)
            reward -= hand_size_penalty
        else:
            hand_size_penalty = reward_hand_size_penalty * (len(self.ai_hand) / 10)
            reward -= hand_size_penalty

        # 过早出牌惩罚（基于游戏阶段）
        if self.current_player == 'player':
            game_stage = (10 - len(self.player_hand)) / 10
        else:
            game_stage = (10 - len(self.ai_hand)) / 10

        action_type = 'pair' if len(action) == 2 and action[0] == action[1] else 'triple' if len(
            action) == 4 and action[:3] == action[0] * 3 else None
        if action_type == 'pair' and game_stage < 0.6:
            reward *= (1 - game_stage * reward_premature_play_penalty)

        # 根据牌数调整奖惩强度
        if self.current_player == 'player':
            opponent_cards = len(self.ai_hand)
            own_cards = len(self.player_hand)
            if opponent_cards < 5:
                reward -= opponent_cards * reward_opponent_low_cards_penalty
            if own_cards < 5:
                reward += (5 - own_cards) * reward_ai_low_cards_bonus
        else:
            opponent_cards = len(self.player_hand)
            own_cards = len(self.ai_hand)
            if opponent_cards < 5:
                reward -= opponent_cards * reward_opponent_low_cards_penalty
            if own_cards < 5:
                reward += (5 - own_cards) * reward_ai_low_cards_bonus

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


# 训练 DQN
def train_dqn(env, num_episodes=num_episodes, initial_gamma=initial_gamma, gamma_decay=gamma_decay,
              initial_epsilon=initial_epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
              learning_rate=learning_rate, batch_size=batch_size, memory_capacity=memory_capacity,
              target_update_freq=target_update_freq):
    print("Training started...")
    input_size = len(env._get_state())
    model = DQN(input_size)
    target_model = DQN(input_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(memory_capacity)

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        current_epsilon = max(epsilon_min, initial_epsilon * (epsilon_decay ** episode))

        while not done:
            if env.current_player == 'player':
                valid_actions = env._get_valid_actions(env.player_hand)
                if not valid_actions:
                    action = 'pass'
                else:
                    action = random.choice(valid_actions)
            else:
                valid_actions = env._get_valid_actions(env.ai_hand)
                if not valid_actions:
                    action = 'pass'
                else:
                    action = random.choice(valid_actions)

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            # 存储经验时去除batch维度
            replay_buffer.push(state.squeeze(0).numpy(), action, reward, next_state.squeeze(0).numpy(), done)

            # 经验回放
            if len(replay_buffer) > batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
                batch_state = torch.tensor(batch_state, dtype=torch.float32)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(1)
                batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(1)

                # 计算目标 Q 值
                with torch.no_grad():
                    next_q_values = target_model(batch_next_state)
                    max_next_q_values = torch.max(next_q_values, dim=1)[0]
                    target_q_values = batch_reward + (1 - batch_done) * initial_gamma * max_next_q_values.unsqueeze(1)

                # 计算当前 Q 值
                current_q_values = model(batch_state)
                action_indices = torch.tensor([ACTION_TO_INDEX[action] for action in batch_action], dtype=torch.long)
                current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

                # 计算损失并更新模型
                loss = criterion(current_q_values, target_q_values.squeeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

            # 更新目标网络
            if episode % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {current_epsilon:.4f}")

    print("Training finished.")
    return model


# 保存模型
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


# 加载模型
def load_model(input_size, model_path):
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == "__main__":
    # 训练模型
    env = CardGameEnv()
    trained_model = train_dqn(env)
    model_path = "models/model" + str(num_episodes) + ".pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(trained_model, model_path)
    print("Model saved successfully.")