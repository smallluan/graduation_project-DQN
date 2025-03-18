import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 全局参数
num_episodes = 100  # 训练轮数
initial_gamma = 0.99  # 初始折扣因子
gamma_decay = 0.999   # 折扣因子衰减率
initial_epsilon = 0.1  # 初始探索率
epsilon_decay = 0.995 # 探索率衰减率
learning_rate = 0.001  # 学习率

# 奖励相关参数
reward_invalid_action = -1  # 无效动作奖励，惩罚加重
reward_pass_with_valid = -1  # 有有效动作时 pass 的奖励，惩罚加重
reward_pass_without_valid = -0.5  # 无有效动作时 pass 的奖励
reward_single_card = 1  # 出单牌奖励，适当提高
reward_pair_card = 1  # 出对子奖励，适当提高
reward_triple_with_single = 3  # 出三带一奖励，适当提高
reward_player_win = -10  # 玩家获胜奖励，惩罚加重
reward_ai_win = 10  # AI 获胜奖励，奖励加重
reward_pair_win_bonus = 5  # 对子获胜额外奖励，提高
reward_split_pair_penalty = -5  # 拆对子惩罚，加重

# 预定义所有可能的动作列表
# 必须与训练模型时的动作列表完全一致！
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
        self.fc1 = nn.Linear(input_size, 1024)  # 增加第一层神经元数量
        self.fc2 = nn.Linear(1024, 1024)         # 增加第二层神经元数量
        self.fc3 = nn.Linear(1024, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义纸牌游戏环境
class CardGameEnv:
    def __init__(self):
        self.reset()
        self.just_got_turn = True  # 标记是否刚获得出牌权
        self.last_played_pair = False  # 标记上一轮是否出的对子
        self.last_played_triple = False  # 标记上一轮是否出的三带一
        self.last_action_was_pass = False  # 标记上一轮是否是 pass

    def reset(self):
        self.deck = []
        for num in range(1, 10):
            self.deck.extend([num] * 4)
        random.shuffle(self.deck)
        self.player_hand = sorted(self.deck[:10])
        self.ai_hand = sorted(self.deck[10:20])
        self.last_played = 0
        self.last_played_pair = False
        self.last_played_triple = False
        self.current_player = 'player'
        self.just_got_turn = True
        self.last_action_was_pass = False
        return self._get_state()

    def _get_state(self):
        ai_hand = self.ai_hand.copy()
        ai_hand += [0] * (10 - len(ai_hand))
        ai_hand = ai_hand[:10]
        state = ai_hand + [len(self.player_hand), self.last_played, int(self.last_played_pair),
                           int(self.last_played_triple)]
        return np.array(state, dtype=np.float32)

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
                print("You just got the turn, you can't pass.")
                reward = reward_invalid_action
            else:
                valid_actions = self._get_valid_actions(self.player_hand)
                if action == 'pass':
                    if valid_actions:
                        print("You have valid cards, you can't pass.")
                        reward = reward_pass_with_valid
                    else:
                        self.last_played = 0
                        self.last_played_pair = False
                        self.last_played_triple = False
                        self.current_player = 'ai'
                        self.just_got_turn = True
                        self.last_action_was_pass = True
                        reward = reward_pass_without_valid
                else:
                    try:
                        if len(str(action)) == 2 and str(action)[0] == str(action)[1]:
                            # 出对子
                            card = int(str(action)[0])
                            if self.player_hand.count(card) < 2:
                                print(f"你没有两张 {card}！")
                                reward = reward_invalid_action
                            elif self.last_played_triple:
                                print("必须出更大的三带一！")
                                reward = reward_invalid_action
                            elif self.last_played_pair and card <= self.last_played:
                                print("必须出更大的对子！")
                                reward = reward_invalid_action
                            elif self.last_played > 0 and card <= self.last_played:
                                print("必须大于上一轮的出牌！")
                                reward = reward_invalid_action
                            else:
                                self.player_hand.remove(card)
                                self.player_hand.remove(card)
                                self.last_played = card
                                self.last_played_pair = True
                                self.last_played_triple = False
                                self.last_action_was_pass = False
                                reward = reward_pair_card
                                done = len(self.player_hand) == 0
                                if done and len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1]:
                                    reward += reward_pair_win_bonus
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(self.player_hand) == 1:
                                    reward += reward_split_pair_penalty
                                if done:
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True
                        elif len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
                            # 出三带一
                            triple_card = int(str(action)[0])
                            single_card = int(str(action)[3])
                            if self.player_hand.count(triple_card) < 3 or single_card not in self.player_hand:
                                print(f"你没有三张 {triple_card} 或者没有 {single_card}！")
                                reward = reward_invalid_action
                            elif self.last_played_triple and triple_card <= self.last_played:
                                print("必须出更大的三带一！")
                                reward = reward_invalid_action
                            elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                                print("上一轮出的是对子或单牌，不能出三带一！")
                                reward = reward_invalid_action
                            else:
                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(triple_card)
                                self.player_hand.remove(single_card)
                                self.last_played = triple_card
                                self.last_played_pair = False
                                self.last_played_triple = True
                                self.last_action_was_pass = False
                                reward = reward_triple_with_single
                                done = len(self.player_hand) == 0
                                if done:
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True
                        else:
                            # 出单牌
                            card = int(action)
                            if card not in self.player_hand:
                                print(f"你没有 {card}！")
                                reward = reward_invalid_action
                            elif self.last_played_triple:
                                print("必须出更大的三带一！")
                                reward = reward_invalid_action
                            elif self.last_played_pair:
                                print("必须出更大的对子！")
                                reward = reward_invalid_action
                            elif card <= self.last_played:
                                print("必须大于上一轮的出牌！")
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
                                elif len(prev_player_hand) == 2 and prev_player_hand[0] == prev_player_hand[1] and len(self.player_hand) == 1:
                                    reward += reward_split_pair_penalty
                                if done:
                                    reward = reward_player_win
                                else:
                                    self.current_player = 'ai'
                                    self.just_got_turn = True
                    except ValueError as e:
                        print(f"输入格式错误: {e}")
                        reward = reward_invalid_action
                    except Exception as e:
                        print(f"发生错误: {str(e)}")
                        reward = reward_invalid_action
        else:
            prev_ai_hand = self.ai_hand.copy()
            valid_actions = self._get_valid_actions(self.ai_hand)
            if action == 'pass':
                if valid_actions:
                    print("AI has valid cards, it can't pass.")
                    reward = reward_pass_with_valid
                else:
                    self.last_played = 0
                    self.last_played_pair = False
                    self.last_played_triple = False
                    self.current_player = 'player'
                    self.just_got_turn = True
                    self.last_action_was_pass = True
                    reward = reward_pass_without_valid
            else:
                if len(action) == 2 and action[0] == action[1]:
                    # 出对子
                    card = int(action[0])
                    if self.ai_hand.count(card) < 2:
                        reward = reward_invalid_action
                    elif self.last_played_triple:
                        print("AI 必须出更大的三带一！")
                        reward = reward_invalid_action
                    elif self.last_played_pair and card <= self.last_played:
                        print("AI 必须出更大的对子！")
                        reward = reward_invalid_action
                    elif self.last_played > 0 and card <= self.last_played:
                        print("AI 必须大于上一轮的出牌！")
                        reward = reward_invalid_action
                    else:
                        self.ai_hand.remove(card)
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = True
                        self.last_played_triple = False
                        self.last_action_was_pass = False
                        reward = reward_pair_card
                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += reward_pair_win_bonus
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward += reward_split_pair_penalty
                        if done:
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True
                elif len(action) == 4 and action[:3] == action[0] * 3:
                    # 出三带一
                    triple_card = int(action[0])
                    single_card = int(action[3])
                    if self.ai_hand.count(triple_card) < 3 or single_card not in self.ai_hand:
                        reward = reward_invalid_action
                    elif self.last_played_triple and triple_card <= self.last_played:
                        print("AI 必须出更大的三带一！")
                        reward = reward_invalid_action
                    elif self.last_played_pair or (self.last_played > 0 and not self.last_played_triple):
                        print("AI 上一轮出的是对子或单牌，不能出三带一！")
                        reward = reward_invalid_action
                    else:
                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(triple_card)
                        self.ai_hand.remove(single_card)
                        self.last_played = triple_card
                        self.last_played_pair = False
                        self.last_played_triple = True
                        self.last_action_was_pass = False
                        reward = reward_triple_with_single
                        done = len(self.ai_hand) == 0
                        if done:
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True
                else:
                    # 出单牌
                    card = int(action)
                    if card not in self.ai_hand:
                        reward = reward_invalid_action
                    elif self.last_played_triple:
                        print("AI 必须出更大的三带一！")
                        reward = reward_invalid_action
                    elif self.last_played_pair:
                        print("AI 必须出更大的对子！")
                        reward = reward_invalid_action
                    elif card <= self.last_played:
                        print("AI 必须大于上一轮的出牌！")
                        reward = reward_invalid_action
                    else:
                        self.ai_hand.remove(card)
                        self.last_played = card
                        self.last_played_pair = False
                        self.last_played_triple = False
                        self.last_action_was_pass = False
                        reward = reward_single_card
                        done = len(self.ai_hand) == 0
                        if done and len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1]:
                            reward += reward_pair_win_bonus
                        elif len(prev_ai_hand) == 2 and prev_ai_hand[0] == prev_ai_hand[1] and len(self.ai_hand) == 1:
                            reward += reward_split_pair_penalty
                        if done:
                            reward = reward_ai_win
                        else:
                            self.current_player = 'player'
                            self.just_got_turn = True

        if not done:
            self.just_got_turn = False

        next_state = self._get_state()
        return next_state, reward, done

    def _get_valid_actions(self, hand):
        valid_actions = []
        if self.last_played_triple:
            valid_actions = self._find_triple_with_single(hand)
            valid_actions = [act for act in valid_actions if int(act[:3][0]) > self.last_played]
        elif self.last_played_pair:
            pairs = self._find_pairs(hand)
            valid_actions = [str(c) * 2 for c in pairs if c > self.last_played]
        else:
            if self.last_played > 0:
                valid_single = [str(c) for c in hand if c > self.last_played]
                valid_actions = valid_single
            else:
                valid_single = [str(c) for c in hand if c > self.last_played]
                valid_triple = [f"{t}{t}{t}{s}" for t in self._find_triples(hand) for s in hand if s != t]
                valid_pair = [str(c) * 2 for c in self._find_pairs(hand) if c > self.last_played]
                valid_actions = valid_single + valid_triple + valid_pair

        # 如果有有效动作，禁止 pass
        if valid_actions:
            valid_actions = [a for a in valid_actions if a != 'pass']
        return valid_actions

# 训练 DQN
def train_dqn(env, num_episodes=num_episodes, initial_gamma=initial_gamma, gamma_decay=gamma_decay,
              initial_epsilon=initial_epsilon, epsilon_decay=epsilon_decay, learning_rate=learning_rate):
    print("Training started...")
    input_size = len(env._get_state())
    model = DQN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    total_rewards = []

    for episode in range(num_episodes):
        current_gamma = initial_gamma
        current_epsilon = initial_epsilon
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
                if random.random() < current_epsilon:
                    valid_actions = env._get_valid_actions(env.ai_hand)
                    if not valid_actions:
                        action = 'pass'
                    else:
                        action = random.choice(valid_actions)
                else:
                    q_values = model(state)
                    valid_actions = env._get_valid_actions(env.ai_hand)
                    valid_action_indices = {action: ACTION_TO_INDEX[action] for action in valid_actions if action in ACTION_TO_INDEX}
                    if not valid_actions:
                        action = 'pass'
                    else:
                        valid_q_values = [q_values[0][valid_action_indices[action]] for action in valid_actions if action in valid_action_indices]
                        action = valid_actions[torch.argmax(torch.tensor(valid_q_values)).item()]

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            target = reward
            if not done:
                target = reward + current_gamma * torch.max(model(next_state)).item()

            q_values = model(state)
            if action in ACTION_TO_INDEX:
                q_value = q_values[0][ACTION_TO_INDEX[action]]
            else:
                pass_index = ACTION_TO_INDEX.get('pass', 0)
                q_value = q_values[0][pass_index]

            loss = criterion(q_value, torch.tensor(target, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            current_gamma *= gamma_decay
            current_epsilon *= epsilon_decay

        total_rewards.append(total_reward)

        print(f"{((episode + 1) / num_episodes) * 100:.2f}%")

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

# 游戏主循环
def play_game(env, model):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False

    while not done:
        if env.current_player == 'player':
            print(f"Your hand: {env.player_hand}")
            print(f"AI hand: {env.ai_hand}")
            if env.last_played_triple:
                print(f"Last played triple with single: {env.last_played}")
            elif env.last_played_pair:
                print(f"Last played pair: {env.last_played}")
            else:
                print(f"Last played: {env.last_played}")
            valid_actions = env._get_valid_actions(env.player_hand)
            print(f"Valid actions: {valid_actions}")
            while True:
                action = input("请输入要出的牌（例如 '3' 出单张，'33' 出对子，'3334' 出三带一，'pass' 过牌）: ").strip().lower()
                if action == 'pass' and valid_actions:
                    print("你有有效的出牌动作，不能 pass，请重新输入。")
                elif action in valid_actions or (action == 'pass' and not valid_actions):
                    break
                else:
                    print(f"无效动作！有效动作：{valid_actions}")
        else:
            # AI 的逻辑
            q_values = model(state)
            valid_actions = env._get_valid_actions(env.ai_hand)
            valid_action_indices = {action: ACTION_TO_INDEX[action] for action in valid_actions if action in ACTION_TO_INDEX}
            if not valid_actions:
                action = 'pass'
            else:
                valid_q_values = [q_values[0][valid_action_indices[action]] for action in valid_actions if action in valid_action_indices]
                action = valid_actions[torch.argmax(torch.tensor(valid_q_values)).item()]

            print(f"AI plays: {action}")

        next_state, reward, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = next_state

        if done:
            if len(env.player_hand) == 0:
                print("You win!")
            else:
                print("AI wins!")

if __name__ == "__main__":
    env = CardGameEnv()
    # 训练模型
    trained_model = train_dqn(env)
    # 保存模型
    model_path = "models/model"+str(num_episodes)+".pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(trained_model, model_path)
    print("Model saved successfully.")

    # 加载模型
    input_size = len(env._get_state())
    loaded_model = load_model(input_size, model_path)

    # 开始游戏
    # while True:
    #     play_game(env, loaded_model)