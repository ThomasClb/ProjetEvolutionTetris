from tetris import Tetris
import random as rd

def main():
    env = Tetris()
    render_delay = 0.1
    for i in range(20):
        next_states = env.get_next_states()
        choosen_action_index = rd.randint(0, len(next_states.items()) - 1)
        i = 0
        best_action = None
        for action, state in next_states.items():
            if i == choosen_action_index:
                best_action = action
                break
            i += 1
        print(best_action)
        reward, done = env.play(best_action[0], best_action[1], render=True, render_delay=render_delay)

if __name__ == '__main__':
    main()
