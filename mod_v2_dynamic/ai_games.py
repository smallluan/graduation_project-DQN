import os
import random

import torch
from main import CardGameEnv, DQN, ACTION_TO_INDEX

l = 10

# 动态加载模型
def load_model(model_path):
    env = CardGameEnv()
    input_size = len(env._get_state())  # 自动获取最新输入维度
    model = DQN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# AI决策函数
def ai_play(model, state, valid_actions):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = model(state_tensor)

    valid_indices = {act: ACTION_TO_INDEX[act] for act in valid_actions if act in ACTION_TO_INDEX}
    if not valid_indices:
        return 'pass'

    valid_q = [q_values[0][valid_indices[act]] for act in valid_actions]
    best_action_idx = torch.argmax(torch.tensor(valid_q)).item()
    return valid_actions[best_action_idx]

def setData(l):
    deck = [i for i in range(1, 10) for _ in range(4)]
    hand_data = []
    for _ in range(l):
        random.shuffle(deck)
        hand_data.append((sorted(deck[:10]), sorted(deck[10:20])))
    with open('data.py', 'w') as f:
        f.write(f"hand_data = {hand_data}")


# 游戏主循环
def main():
    model1_path = "models/model100.pth"  # 模型路径需与训练保存一致
    model2_path = "models/model50.pth"
    # 1. 未对 ai 出牌进行限制 ，可能过早出较大的牌时（双方都是）
    # 2625 ： 2375 ->  100 ： 50
    # 2535 ： 2465 ->  100 : 50 (换牌)

    # 2551 : 2449 ->  1000 : 100
    # 2951 ： 2049 ->  1000 : 100(换牌)

    # 2441 : 2559 ->  1000 : 50
    # 2766 : 2234 ->  1000 : 50(换牌)

    # 2. 对 50 的 ai 进行限制，防止过早出较大的牌
    # 2090 ： 2910 ->  100 ： 50
    # 2270 ： 2730 ->  100 ： 50(换牌)

    # 3. 对 50 和 100 的 ai 进行限制，防止过早出较大的牌
    # 2213 ： 2787 ->  100 ： 50
    # 2477 ： 2523 ->  100 ： 50(换牌)
    # // 100 比不过 50 的可能是训练过少导致随机性大于训练的导向性，因为下面 1000 训练量的 ai 与 50 的对比就可以证明这一点

    # 3. 对 50 和 1000 的 ai 进行限制，防止过早出较大的牌
    # 2855 ： 2145 ->  1000 ： 50
    # 3340 ： 1660 ->  1000 ： 50(换牌)

    # 4. 对 1000 的 ai 进行限制，防止拆开对子或者三代一
    # 2966 ： 2034 ->  1000 ： 50
    # 2610 ： 2390 ->  1000 ： 50(换牌)

    model1 = load_model(model1_path)  # 玩家1使用的模型
    model2 = load_model(model2_path)  # 玩家2使用的模型

    stats = {
        'player1_wins': 0,
        'player2_wins': 0,
        'draws': 0,
        'rounds': []
    }

    for idx, (p1_hand, p2_hand) in enumerate(hand_data[:l], 1):
        env = CardGameEnv()
        # 使用测试数据初始化环境
        env.reset(
            player_hand=p1_hand,
            ai_hand=p2_hand,
            starting_player='player'  # 起始玩家可以是 'player' 或 'ai'
        )
        round_records = []

        print(f"\n第 {idx} 局开始")
        print(f"玩家1手牌: {p1_hand}")
        print(f"玩家2手牌: {p2_hand}")

        while True:
            state = env._get_state()
            current_player = env.current_player

            # 获取有效动作
            if current_player == 'player':
                valid_actions = env._get_valid_actions(env.player_hand)
                current_model = model1
                player_id = 'player1'
            else:
                valid_actions = env._get_valid_actions(env.ai_hand)
                current_model = model2
                player_id = 'player2'

            # AI决策
            action = ai_play(current_model, state, valid_actions)
            round_records.append((player_id, action))

            if action != 'pass':
                if int(action) > 1000:
                    print(idx, player_id, action)

            print(f"{player_id} 出牌: {action}")

            # 执行动作
            next_state, reward, done = env.step(action)

            if done:
                if current_player == 'player':
                    stats['player1_wins'] += 1
                else:
                    stats['player2_wins'] += 1
                stats['rounds'].append({
                    'index': idx,
                    'p1_hand': p1_hand,
                    'p2_hand': p2_hand,
                    'moves': round_records,
                    'winner': player_id
                })
                # 打印当前进度和胜负统计
                print(f"第 {idx} 局已完成，玩家1获胜次数: {stats['player1_wins']}，玩家2获胜次数: {stats['player2_wins']}")
                # print(f"本局获胜者: {player_id}")
                break

    # 打印统计结果
    print("\n=== 博弈统计结果 ===")
    print(f"总轮数: {l}")
    print(f"玩家1胜利次数: {stats['player1_wins']}")
    print(f"玩家2胜利次数: {stats['player2_wins']}")

    # 保存详细记录（可选）
    # import json
    # with open('game_records.json', 'w') as f:
    #     json.dump(stats['rounds'], f, indent=2)
    # print("\n详细记录已保存到 game_records.json")


if __name__ == "__main__":
    if not os.path.exists('test_data/data.py'):
        setData(l)
    from test_data.data import hand_data
    main()