import os
import torch
import torch.nn as nn
import random

# 全局参数
l = 1000  # 对局次数

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


# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, OUTPUT_SIZE)  # 修改输出层大小

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
        print(f"本局开始，玩家1手牌: {self.player1_hand}，玩家2手牌: {self.player2_hand}")
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
            print(f"{self.current_player} 刚拿到出牌权，不能 pass！")
            return self._get_state(), done
        elif action == 'pass':
            if valid_actions == ['pass']:
                # 如果有效动作只有 pass，允许 pass
                self.last_played = 0
                self.last_played_pair = False
                self.last_played_triple = False  # 重置状态
                next_player = 'player2' if self.current_player == 'player1' else 'player1'
                print(f"{self.current_player} 无有效出牌，选择过牌，当前玩家切换为 {next_player}")
                self.current_player = next_player
                self.just_got_turn = True
            else:
                print(f"{self.current_player} 有有效出牌，不能 pass！")
                return self._get_state(), done
        else:
            if action not in valid_actions:
                print(f"{self.current_player} 出牌 {action} 无效！")
                return self._get_state(), done

            try:
                if len(str(action)) == 2 and str(action)[0] == str(action)[1]:
                    # 出对子
                    card = int(str(action)[0])
                    current_hand.remove(card)
                    current_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = True
                    self.last_played_triple = False
                    done = len(current_hand) == 0
                    next_player = 'player2' if self.current_player == 'player1' else 'player1'
                    print(f"{self.current_player} 出对子 {action}，当前玩家切换为 {next_player}")
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
                elif len(str(action)) == 4 and str(action)[:3] == str(action)[0] * 3:
                    # 出三带一
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
                    print(f"{self.current_player} 出三带一 {action}，当前玩家切换为 {next_player}")
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
                else:
                    # 出单牌
                    card = int(action)
                    current_hand.remove(card)
                    self.last_played = card
                    self.last_played_pair = False
                    self.last_played_triple = False
                    done = len(current_hand) == 0
                    next_player = 'player2' if self.current_player == 'player1' else 'player1'
                    print(f"{self.current_player} 出单张 {action}，当前玩家切换为 {next_player}")
                    if not done:
                        self.current_player = next_player
                        self.just_got_turn = True
            except:
                print(f"{self.current_player} 出牌格式错误！")
                return self._get_state(), done

        if not done:
            self.just_got_turn = False

        next_state = self._get_state()
        print(f"当前玩家1手牌: {self.player1_hand}，玩家2手牌: {self.player2_hand}")
        return next_state, done

    def get_valid_actions(self, hand):
        valid_actions = []
        if self.last_played == 0:
            # 对方 pass 或者开局，允许出任意类型的牌
            valid_single = [str(c) for c in hand]
            valid_actions = valid_single

            # 添加对子和三带一
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
            # 只考虑单牌
            valid_single = [str(c) for c in hand if c > self.last_played]
            valid_actions = valid_single

        # 如果刚拿到出牌权，禁止 pass
        if self.just_got_turn:
            valid_actions = [a for a in valid_actions if a != 'pass']

        # 如果无有效动作，允许 pass
        if not valid_actions:
            valid_actions = ['pass']

        return valid_actions


# 加载模型
def load_model(input_size, model_path):
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 进行对局
def play_games(env, model1, model2):
    m1_wins = 0
    m2_wins = 0
    input_size = len(env._get_state()[0])

    for i in range(l):
        print(f"\n第 {i + 1} 局开始")
        player1_hand, player2_hand = hand_data[i]
        state = env.reset(player1_hand, player2_hand)
        done = False

        while not done:
            current_model = model1 if env.current_player == 'player1' else model2
            q_values = current_model(state)

            current_hand = env.player1_hand if env.current_player == 'player1' else env.player2_hand
            valid_actions = env.get_valid_actions(current_hand)

            print(f"当前玩家: {env.current_player}, 有效动作: {valid_actions}")

            # 映射到预定义动作的索引
            valid_indices = [ACTION_TO_INDEX[action] for action in valid_actions if action in ACTION_TO_INDEX]
            if not valid_indices:
                action = 'pass'
            else:
                valid_q = q_values[0][valid_indices]
                best_idx = torch.argmax(valid_q).item()
                action = valid_actions[best_idx]

            next_state, done = env.step(action)
            state = next_state

        if len(env.player1_hand) == 0:
            m1_wins += 1
            print(f"第 {i + 1} 局结束，玩家1获胜")
        else:
            m2_wins += 1
            print(f"第 {i + 1} 局结束，玩家2获胜")
        print(f"模型 m1 胜场: {m1_wins}")
        print(f"模型 m2 胜场: {m2_wins}")


# 生成测试数据
def setData(l):
    deck = [i for i in range(1, 10) for _ in range(4)]
    hand_data = []
    for _ in range(l):
        random.shuffle(deck)
        hand_data.append((sorted(deck[:10]), sorted(deck[10:20])))
    with open('data.py', 'w') as f:
        f.write(f"hand_data = {hand_data}")


if __name__ == "__main__":
    # 初始化环境
    if not os.path.exists('data.py'):
        setData(l)
    from data import hand_data

    # 示例初始化
    env = CardGameEnv([], [])  # 手牌通过reset设置
    input_size = len(env._get_state()[0])

    # 加载模型（需确保模型使用新动作列表训练）
    model1 = load_model(input_size, "models/model50.pth")
    model2 = load_model(input_size, "models/model10000.pth")

    play_games(env, model1, model2)
