
# BC_train.py：使用行為複製的預訓練程式碼。如果 CUDA 可用，訓練預設在 GPU 上運行，也可以在 CPU 上訓練，但速度要慢得多。

import numpy as np
import torch
from Net import ACNetwork # 導入神經網絡模型
from routing_gym import RoutingEnv # 導入環境
from IL_expert_alternate import Expert, generate_coordinates, switcher # 導入專家類和輔助函數
from PPO_structure import PPO # 導入PPO類


if __name__ == "__main__":
    expert = Expert() # 實例化專家類
    num_agents = 5 # 環境中的智能體數量

    # 生成環境的起始和終止位置
    start_pos, end_pos = generate_coordinates() # 生成隨機座標點
    env = RoutingEnv(start_pos, end_pos) # 創建環境對象
    max_step = env.wsize ** 2 # 每個回合的最大步數（基於環境大小）

    device = torch.device('cuda') # 如果可用，使用GPU
    PPO_model = ACNetwork().to(device) # 創建神經網絡模型
    #PPO_model.load_state_dict(torch.load('IL_agent.pt')) # 加載預訓練模型權重

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
    # model 模型初始化
    # ----------------------------------------- #

    # 創建集中式PPO代理
    centralized_agent = PPO(0, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma,
                            device)
     # 為環境中的每個智能體創建個別的PPO代理
    PPO_agent = [PPO(i + 1, PPO_model, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device) for i in
                 range(num_agents)]
    # ----------------------------------------- #
    # 訓練--回合更新 on_policy
    # ----------------------------------------- #

    # 生成每個回合的起始和終止位置，並確保沒有捷徑
    start_pos, end_pos = generate_coordinates()
    start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
    while shorts:
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

    print(start_pos, end_pos)

    # 迭代各個回合
    for i in range(num_episodes):
        # 生成當前回合的起始和終止位置
        start_pos, end_pos = generate_coordinates()
        start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)
        while shorts:
            start_pos, end_pos = generate_coordinates()
            start_pos, end_pos, shorts = expert.instruct(start_pos, end_pos)

        # 重置當前回合的環境
        state = env.reset(start_pos, end_pos)  # 環境重製
        done = False
        episode_return, executing_id, num_step = 0, 0, 0  # 總reward, 正在執行動作的agent ID # 初始化回合變量
        # 儲存每個episode的數據
        transition_dict = {
            'observations': [],
            'distances': [],
            'actions': [],
            'positions': [],
        }

        # 執行回合直到終止或達到最大步數
        while not done and num_step < max_step:
            executing_id = switcher(executing_id, num_agents, state) # 切換執行智能體的ID
            #action_p = PPO_agent[executing_id-1].deterministic_action(state)  # 確定動作概率
            action = expert.policy[num_step] # 從專家策略獲取動作
            next_state, reward, done, _ = env.step(executing_id, action)  # 环境更新 # 在環境中執行動作
            # 收集轉換數據
            transition_dict['observations'].append(state.getObservation(executing_id))
            transition_dict['distances'].append(state.getDistance(executing_id))
            transition_dict['positions'].append(state.getPos(executing_id))
            transition_dict['actions'].append(action)
            # 更新狀態和步數計數
            state = next_state
            num_step += 1
            # 累績獎勵

        # 模型訓練 # 使用收集的轉換數據訓練集中式PPO代理
        for key in transition_dict:
            transition_dict[key] = np.array(transition_dict[key])

        print('Episode', i, ':', end='')
        centralized_agent.imitate(transition_dict)
        # 定期保存模型權重
        if i % 100 == 0:
            torch.save(PPO_model.state_dict(), 'IL_agent.pt')
