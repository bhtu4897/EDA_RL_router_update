
# PPO_train.py：訓練程式碼。如果 CUDA 可用，訓練預設在 GPU 上運行，也可以在 CPU 上訓練，但速度要慢得多。
#這段程式碼是一個訓練迴圈，用於在環境中訓練智能體，並且使用逐步更新的方式進行

import numpy as np
import torch
from Net import ACNetwork  # 從Net.py中導入ACNetwork模型
from routing_gym import RoutingEnv  # 導入自定義的RoutingEnv環境類
from IL_expert_alternate import Expert, generate_coordinates, switcher  # 導入專家、生成坐標的函數和switcher函數
from PPO_structure import PPO  # 導入PPO算法的實現


if __name__ == "__main__":
    expert = Expert() # 創建專家對象
    num_agents = 5  # 智能體的數量

    start_pos, end_pos = generate_coordinates()  # 生成起點和終點坐標
    env = RoutingEnv(start_pos, end_pos)  # 創建環境對象
    max_step = env.wsize ** 2  # 最大步數，即最大可移動次數

    device = torch.device('cuda')  # 指定設備為GPU
    PPO_model = ACNetwork().to(device)  # 創建ACNetwork模型並移至GPU
    PPO_model.load_state_dict(torch.load('IL_agent.pt'))  # 從IL_agent.pt中載入模型參數

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
    # model
    # ----------------------------------------- #

    centralized_agent = PPO(0, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                            device) # 創建中心化訓練的PPO智能體
    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in
                 range(num_agents)] # 創建分散式訓練的PPO智能體

    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    # done = False
    # start_pos, end_pos = generate_coordinates()
    # start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
    # while shorts:
    #     start_pos, end_pos = generate_coordinates()
    #     start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

    for i in range(num_episodes):

        done = False # 標記是否完成任務 
        start_pos, end_pos = generate_coordinates() # 生成新的起點和終點坐標
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)  # 使用專家對坐標進行指導
        while shorts:   # 如果存在shorts（需要重新指導）
            start_pos, end_pos = generate_coordinates() # 重新生成坐標
            start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos) # 使用專家對坐標進行指導

        state = env.reset(start_pos, end_pos)  # 環境重製
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID,步數
        
        # 儲存每個episode的數據
        transition_dict = {
            'observations': [],
            'distances': [],
            'actions': [],
            'positions': [],
            'next_observations': [],
            'next_distances': [],
            'next_positions': [],
            'rewards': [],
            'dones': [],
        }
        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state)  # 切換執行的智能體ID
            if i % 100 < 80:
                action = PPO_agent[executing_id-1].stochastic_action(state)  # 使用PPO智能體進行隨機策略
            else:
                action = expert.policy[num_step]  # 使用專家的策略
            next_state, reward, done, _ = env.step(executing_id, action)  # 進行一步環境更新
            # 儲存transition數據
            transition_dict['observations'].append(state.getObservation(executing_id))
            transition_dict['distances'].append(state.getDistance(executing_id))
            transition_dict['positions'].append(state.getPos(executing_id))
            transition_dict['actions'].append(action)
            transition_dict['next_observations'].append(next_state.getObservation(executing_id))
            transition_dict['next_distances'].append(next_state.getDistance(executing_id))
            transition_dict['next_positions'].append(next_state.getPos(executing_id))
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新狀態和步數
            state = next_state
            num_step += 1
            # 累計獎勵
            episode_return += reward



        # 模型訓練
        for key in transition_dict:
            transition_dict[key] = np.array(transition_dict[key])

        centralized_agent.learn(transition_dict)


        print('Episode', i, ':', episode_return, 'Done: ', done)
        if i % 100 == 0 and i != 0:  # 每100個回合保存模型參數
            torch.save(PPO_model.state_dict(), 'RL_agent.pt') 
