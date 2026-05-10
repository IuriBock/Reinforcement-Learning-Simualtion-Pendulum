import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

try:
    # Configuração do Ambiente
    env = gym.make("CartPole-v1", render_mode="human")
    env.metadata['render_fps'] = 10000 # Definindo quantidade de frames por segundo

    # Discretização: Transformar valores contínuos em categorias (bins)
    # Ângulo da Barra e Velocidade Angular
    def discretize_state(state):
        # posição, vel, ângulo, vel_angular
        _, _, angle, angular_vel = state
        angle_bin = int(np.digitize(angle, np.linspace(-0.204, 0.204, 20)))
        vel_bin = int(np.digitize(angular_vel, np.linspace(-3.0, 3.0, 20)))
        return (angle_bin, vel_bin)

    #  Inicialização da Q-Table (O "Cérebro" da IA)
    # Uma matriz 11x11 para os estados e 2 para as ações (esquerda/direita)
    q_table = np.zeros((21, 21, env.action_space.n))

    learning_rate = 0.03
    discount_factor = 0.95
    epsilon = 1.0  # Exploração inicial (100% aleatório)  
    days = 10000

    # --- CONFIGURAÇÃO DO GRÁFICO ---
    plt.ion() # Liga o modo interativo
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], color="#1D08DA", label='Tempo Médio')
    ax.set_xlim(0, days) # Total de episódios
    ax.set_ylim(0, 510) 
    ax.set_title('Telemetria de Aprendizado em Tempo Real')
    
    ax2 = ax.twinx() # Cria um segundo eixo Y à direita para as recompensas
    line_reward, = ax2.plot([], [], color="#058B00", label='Recompensa Média', alpha=0.6)
    
    ax2.set_ylabel('Recompensa')
    ax.set_xlabel('Episódio')
    ax.set_ylabel('Tempo Média')

    lines = [line, line_reward]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    plt.style.use('dark_background') 

    rewards_history = []
    mean_rewards = []
    frames_history = []
    mean_frames = []

    # Loop de Treinamento
    for day in range(days):
        state, _ = env.reset()
        state_adj = discretize_state(state)
        done = False
        frames = 0
        total_episode_reward = 0
        while not done:

            # Decisão: Explorar (aleatório) ou Explorar (usar o que aprendeu)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_adj])

            # reward = 1 para cada frame 
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_adj = discretize_state(next_state)

            x_pos = next_state[0]
            angle = next_state[2]

            # Quanto menor o x_pos absoluto, maior o bônus.
            # Se x_pos for 0 (centro), ganha +1 extra.
            center_reward = 1.0 - abs(x_pos / 2.4) 

            # Bônus por manter o poste bem vertical:
            angle_reward = 1.0 - abs(angle / 0.204)

            # Recompensa Final Combinada
            custom_reward = (reward * 0.8) + (center_reward * 0.5) + (angle_reward * 0.2)

            # Se ele cair ou sair do limite
            if terminated:
                if frames >= 500:
                    custom_reward = 100
                else:
                    custom_reward = -700
            print(custom_reward)    

            # A Mágica: Atualização da Q-Table (Equação de Bellman simplificada)
            old_value = q_table[state_adj][action]
            next_max = np.max(q_table[next_state_adj])
            # Novo valor = valor antigo + erro de aprendizado
            q_table[state_adj][action] = old_value + learning_rate * (custom_reward + discount_factor * next_max - old_value)
            
            state_adj = next_state_adj
            frames += 1
            total_episode_reward += custom_reward

        # Diminuir a aleatoriedade conforme a IA aprende
        epsilon = max(0.01, epsilon * 0.99)
        frames_history.append(frames)
        rewards_history.append(total_episode_reward)
        # Calcula a média dos últimos 10 episódios para o gráfico não oscilar demais
        if day > 10:
            current_mean_frames = np.mean(frames_history[-10:])
            current_mean_reward = np.mean(rewards_history[-10:])

            mean_frames.append(current_mean_frames)
            mean_rewards.append(current_mean_reward)

            # Atualiza os dados do gráfico
            line.set_xdata(range(len(mean_frames)))
            line.set_ydata(mean_frames)

            line_reward.set_xdata(range(len(mean_rewards)))
            line_reward.set_ydata(mean_rewards)

            ax2.relim()
            ax2.autoscale_view()
            
            # Atualiza o desenho da janela
            plt.draw()
            plt.pause(0.001) # Pequena pausa para o SO processar o desenho

        if day % 100 == 0:
            print(f"Episódio: {day} - IA evoluindo...")


except KeyboardInterrupt:
    pass

finally:
    env.close()