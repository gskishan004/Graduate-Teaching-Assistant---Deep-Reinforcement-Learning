import gym
from collections import deque
from dqn import dqn
from utils import process
from utils import generate

rendering = True
episode_start = 1
episode_end = 600
batch_size = 64

if __name__ == '__main__':

    env = gym.make('CarRacing-v0')
    agent = dqn()

    for e in range(episode_start, episode_end + 1):
        init_state = env.reset()
        init_state = process(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False
        rewards = []
        total_orig_reward = 0
        while True:
            if rendering:
                env.render()

            current_state_frame_stack = generate(state_frame_stack_queue)
            action = agent.act(current_state_frame_stack)

            reward = 0
            reward_orig = 0
            for _ in range(2 + 1):
                next_state, r, done, info = env.step(action)
                reward += r
                reward_orig += r
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            # Extra bonus for the model if it uses full gas
            if action[1] == 1 and action[2] == 0:
                reward *= 1.5

            total_reward += reward
            total_orig_reward += reward_orig

            next_state = process(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate(state_frame_stack_queue)

            agent.memo(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done or negative_reward_counter >= 25 or total_reward < 0:
                print(
                    'Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2}, Epsilon: {:.2}'.format(e,
                                                                                                                     episode_end,
                                                                                                                     time_frame_counter,
                                                                                                                     float(
                                                                                                                         total_reward),
                                                                                                                     float(
                                                                                                                         agent.epsilon)))
                break
            if len(agent.memory) > batch_size:
                agent.replay_buffer(batch_size)
            time_frame_counter += 1

        if e % 5 == 0:
            agent.update_target()

        if e % 25 == 0:
            agent.save('./save/trial_{}.h5'.format(e))

        print(total_orig_reward)
        rewards.append(total_orig_reward)

    env.close()
