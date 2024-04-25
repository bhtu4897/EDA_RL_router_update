
# PPO_test.py：測試程式碼。結果可以在網格地圖上可視化。
#這段程式主要是使用 PPO 算法訓練分散式智能體在給定環境下完成任務。

import torch
from Net import ACNetwork  # 匯入網路模型
from routing_gym import RoutingEnv  # 匯入環境模型
from IL_expert_alternate import Expert, generate_coordinates, RoutePainter, switcher  # 匯入專家、座標生成器、路線繪製器、切換器
from PPO_structure import PPO  # 匯入PPO結構




if __name__ == "__main__":
    expert = Expert()  # 初始化專家
    num_agents = 5  # 智能體數量

    # 生成起始和終點座標
    start_pos, end_pos = generate_coordinates()
    env = RoutingEnv(start_pos, end_pos) # 初始化環境
    max_step = env.wsize ** 2 # 最大步數

    device = torch.device('cuda')  # 使用GPU
    PPO_model = ACNetwork().to(device)  # 初始化PPO模型
    PPO_model.load_state_dict(torch.load('RL_agent.pt'))  # 載入預訓練參數

    # ----------------------------------------- #
    # 參數設置
    # ----------------------------------------- #

    num_episodes = 100000  # 迭代次數
    actor_lr = 2e-5  # 策略網路學習率
    critic_lr = 2e-5  # 價值網路學習率
    lmbda = 0.95  # 優勢函數系數
    epochs = 10  # 一個批次訓練的次數
    eps = 0.2  # PPO中限制更新範圍參數
    gamma = 0.95  # 折扣因子

    # ----------------------------------------- #
    # model PPO智能體
    # ----------------------------------------- #

    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in range(num_agents)]  # decentralized execution

    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    done = False
    i = 0
    while not done:
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos) # 獲取專家指導
        while shorts:
            start_pos, end_pos = generate_coordinates()
            start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos) # 確保沒有"shorts"
        state = env.reset(start_pos, end_pos)  # 環境重置
        done = False  # 任務完成
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID

        print(start_pos, end_pos)

        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state)  # 切換智能體
            #action = expert.policy[num_step]
            action = PPO_agent[executing_id-1].stochastic_action(state)  # 動作選擇
            next_state, reward, done, _ = env.step(executing_id, action)  # 環境更新
            state = next_state
            num_step += 1
            episode_return += reward

        print('Episode', i, 'Total_reward:', episode_return, ', Done: ', done)
        if done:
            RoutePainter(20, state.paths)  # 繪製路徑
        i += 1



