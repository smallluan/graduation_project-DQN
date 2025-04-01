import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 全局参数
learning_rate = 0.001
batch_size = 64
memory_capacity = 10000
target_update_freq = 500
online_epsilon = 0.1

# 动作定义
ALL_ACTIONS = [
    'pass',
    *[str(i) for i in range(1, 10)],
    *[f"{i}{i}" for i in range(1, 10)],
    *[f"{i}{i}{i}{j}" for i in range(1, 10) for j in range(1, 10) if j != i]
]
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}
OUTPUT_SIZE = len(ALL_ACTIONS)


# DQN网络定义
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


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return tuple(zip(*samples))

    def __len__(self):
        return len(self.buffer)


# 游戏环境
class CardGameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        deck = []
        for num in range(1, 10):
            deck.extend([num] * 4)
        random.shuffle(deck)
        self.player_hand = sorted(deck[:10])
        self.ai_hand = sorted(deck[10:20])
        self.last_played = 0
        self.last_played_pair = False
        self.last_played_triple = False
        self.current_player = 'player'
        self.last_action_was_pass = True  # 游戏开始，视为上一轮是 pass
        return self._get_state()

    # 修改后的 CardGameEnv 的 _get_state 方法
    def _get_state(self):
        player_counts = [self.player_hand.count(i) for i in range(1, 10)]
        ai_counts = [self.ai_hand.count(i) for i in range(1, 10)]
        state = np.concatenate([
            ai_counts, player_counts,
            [len(self.player_hand) / 10, len(self.ai_hand) / 10],
            [self.last_played / 9, int(self.last_played_pair), int(self.last_played_triple)]
        ])
        return state.astype(np.float32)

    def _get_valid_actions(self, hand):
        valid_actions = []

        # 上一轮是 pass，当前玩家不能 pass 且可出任意牌
        if self.last_action_was_pass:
            valid_single = [str(c) for c in hand]
            valid_pair = [f"{c}{c}" for c in self._find_pairs(hand)]
            valid_triple = []
            triples = self._find_triples(hand)
            for triple in triples:
                remaining_hand = [card for card in hand if card != triple]
                for single in remaining_hand:
                    valid_triple.append(f"{triple}{triple}{triple}{single}")
            valid_actions = valid_single + valid_pair + valid_triple
        else:
            # 根据上一轮出牌类型生成有效动作
            if self.last_played_triple:
                valid_actions = self._find_triple_with_single(hand)
                valid_actions = [act for act in valid_actions if int(act[:3][0]) > self.last_played]
                valid_actions = [act for act in valid_actions if len(act) == 4 and act[:3] == act[0] * 3]
                valid_actions.append('pass')
            elif self.last_played_pair:
                pairs = self._find_pairs(hand)
                valid_actions = [f"{c}{c}" for c in pairs if c > self.last_played]
                valid_actions = [act for act in valid_actions if len(act) == 2 and act[0] == act[1]]
                valid_actions.append('pass')
            else:
                if self.last_played > 0:
                    valid_actions = [str(c) for c in hand if c > self.last_played]
                    valid_actions = [act for act in valid_actions if len(act) == 1]
                    valid_actions.append('pass')
                else:
                    valid_single = [str(c) for c in hand]
                    valid_pair = [f"{c}{c}" for c in self._find_pairs(hand)]
                    valid_triple = []
                    triples = self._find_triples(hand)
                    for triple in triples:
                        remaining_hand = [card for card in hand if card != triple]
                        for single in remaining_hand:
                            valid_triple.append(f"{triple}{triple}{triple}{single}")
                    valid_actions = valid_single + valid_pair + valid_triple

        # 如果没有合法动作，强制 pass
        if not valid_actions:
            valid_actions = ['pass']

        return valid_actions

    def _find_pairs(self, hand):
        pairs = []
        for i in range(len(hand) - 1):
            if hand[i] == hand[i + 1]:
                pairs.append(hand[i])
        return pairs

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
        reward = 0  # 默认奖励为 0

        if self.current_player == 'player':
            valid_actions = self._get_valid_actions(self.player_hand)
            if action not in valid_actions:
                raise ValueError("Invalid action!")

            if action == 'pass':
                self.last_played = 0
                self.last_played_pair = False
                self.last_played_triple = False
                self.current_player = 'ai'
                self.last_action_was_pass = True
            else:
                if len(action) == 2 and action[0] == action[1]:
                    card = int(action[0])
                    self.player_hand.remove(card)
                    self.player_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = True
                    self.last_played_triple = False
                    self.last_action_was_pass = False
                elif len(action) == 4 and action[:3] == action[0] * 3:
                    triple_card = int(action[0])
                    single_card = int(action[3])
                    self.player_hand.remove(triple_card)
                    self.player_hand.remove(triple_card)
                    self.player_hand.remove(triple_card)
                    self.player_hand.remove(single_card)
                    self.last_played = triple_card
                    self.last_played_pair = False
                    self.last_played_triple = True
                    self.last_action_was_pass = False
                else:
                    card = int(action)
                    self.player_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = False
                    self.last_played_triple = False
                    self.last_action_was_pass = False

                done = len(self.player_hand) == 0
                if done:
                    self.current_player = 'ai'
                    self.last_action_was_pass = True
                else:
                    self.current_player = 'ai'
                    self.last_action_was_pass = False

        else:
            valid_actions = self._get_valid_actions(self.ai_hand)
            if action not in valid_actions:
                raise ValueError("Invalid action!")

            if action == 'pass':
                self.last_played = 0
                self.last_played_pair = False
                self.last_played_triple = False
                self.current_player = 'player'
                self.last_action_was_pass = True
            else:
                if len(action) == 2 and action[0] == action[1]:
                    card = int(action[0])
                    self.ai_hand.remove(card)
                    self.ai_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = True
                    self.last_played_triple = False
                    self.last_action_was_pass = False
                elif len(action) == 4 and action[:3] == action[0] * 3:
                    triple_card = int(action[0])
                    single_card = int(action[3])
                    self.ai_hand.remove(triple_card)
                    self.ai_hand.remove(triple_card)
                    self.ai_hand.remove(triple_card)
                    self.ai_hand.remove(single_card)
                    self.last_played = triple_card
                    self.last_played_pair = False
                    self.last_played_triple = True
                    self.last_action_was_pass = False
                else:
                    card = int(action)
                    self.ai_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = False
                    self.last_played_triple = False
                    self.last_action_was_pass = False

                done = len(self.ai_hand) == 0
                if done:
                    self.current_player = 'player'
                    self.last_action_was_pass = True
                else:
                    self.current_player = 'player'
                    self.last_action_was_pass = False

        next_state = self._get_state()
        return next_state, reward, done


# 初始化环境
env = CardGameEnv()
input_size = len(env.reset())

# 检查模型文件是否存在
model_path = "models/model_teach.pth"
os.makedirs("models", exist_ok=True)
if os.path.exists(model_path):
    print("Loading existing model...")
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
else:
    print("No existing model found. Starting training from scratch.")
    model = DQN(input_size)

target_model = DQN(input_size)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(memory_capacity)

# 训练计数器
train_counter = 0

# 主训练循环
while True:
    print("\n=== New Game ===")
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    game_result = None

    while not done:
        # 玩家回合
        print(f"\nYour hand: {env.player_hand}")
        print(f"\nAi hand: {env.ai_hand}")
        valid_actions = env._get_valid_actions(env.player_hand)
        print("\nour valid_actions: ", valid_actions)
        while True:
            action = input("\nEnter your action: ")
            if action in valid_actions:
                break
            print("Invalid action. Try again.")

        # 执行玩家动作
        next_state, _, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = next_state

        if done:
            game_result = "player_win"
            print("You win!")
            break

        # AI回合
        with torch.no_grad():
            q_values = model(state)
        valid_actions = env._get_valid_actions(env.ai_hand)
        print("ai valid_actions: ", valid_actions)
        valid_indices = [ACTION_TO_INDEX[a] for a in valid_actions if a in ACTION_TO_INDEX]

        if not valid_actions:
            ai_action = 'pass'
        else:
            if random.random() < online_epsilon:
                ai_action = random.choice(valid_actions)
            else:
                valid_q = q_values[0][valid_indices]
                ai_action = valid_actions[torch.argmax(valid_q).item()]

        # 执行AI动作
        next_state, _, done = env.step(ai_action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # 显示AI动作
        print(f"AI played: {ai_action}")

        # 立即收集用户反馈
        while True:
            try:
                reward = float(input("Enter your rating (-10-10): "))
                if -10 <= reward <= 10:
                    break
            except ValueError:
                pass
        while True:
            suggestion = input("Enter suggested action: ")
            if suggestion in ALL_ACTIONS:
                break

        # 如果是最后一步，等待游戏结束判断结果添加奖励
        if done:
            if len(env.ai_hand) == 0:
                game_result = "ai_win"
                print("AI wins!")
                reward += 100
            else:
                game_result = "player_win"
                print("You win!")
                reward -= 100

        # 将用户反馈转化为训练信号
        suggestion_idx = ACTION_TO_INDEX[suggestion]
        with torch.no_grad():
            target_q = torch.zeros(OUTPUT_SIZE)
            target_q[suggestion_idx] = reward
            if not done:
                target_q += 0.9 * torch.max(target_model(next_state))

        # 更新经验回放
        replay_buffer.push(
            state.squeeze().numpy(),
            ai_action,
            reward,
            next_state.squeeze().numpy(),
            done
        )

        state = next_state

    # 进行在线训练
    for _ in range(1):
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            # 计算目标Q值
            with torch.no_grad():
                target_q = rewards + (1 - dones) * 0.9 * torch.max(target_model(next_states), dim=1)[0].unsqueeze(1)

            # 计算当前Q值
            current_q = model(states)
            action_indices = torch.tensor([ACTION_TO_INDEX[a] for a in actions], dtype=torch.long)
            current_q = current_q.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            # 更新模型
            loss = criterion(current_q, target_q.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 更新目标网络
    train_counter += 1
    if train_counter % target_update_freq == 0:
        target_model.load_state_dict(model.state_dict())
        print("Target network updated")

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print("Model saved as models/model_teach.pth")

    # 询问是否继续
    if input("Continue training? (y/n): ").lower() != 'y':
        break