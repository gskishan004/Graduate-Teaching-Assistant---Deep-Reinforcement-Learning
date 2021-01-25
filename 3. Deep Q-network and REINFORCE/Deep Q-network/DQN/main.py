from collections import deque
import argparse
import gym

from dqn import dqn
from utils import generate
from utils import process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()
    train_model = args.model
    play_episodes = 15

    env = gym.make('CarRacing-v0')
    agent = dqn(epsilon=0)
    agent.load(train_model)

    for e in range(play_episodes):
        init_st = env.reset()
        init_st = process(init_st)

        total_rew = 0
        pun_cnt = 0
        stack_frame = deque([init_st] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_cnt = 1
        
        while True:
            env.render()

            current_state_frame_stack = generate(stack_frame)
            action = agent.act(current_state_frame_stack)
            next_state, reward, done, _ = env.step(action)

            total_rew += reward

            next_state = process(next_state)
            stack_frame.append(next_state)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards: {:.2}'.format(e + 1, play_episodes, time_cnt, float(total_rew)))
                break
            time_cnt += 1
