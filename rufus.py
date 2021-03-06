import argparse

from vizdoom import *

from utils import *
from log import Log
from agent import RdqnAgent

import math
from random import random, randint

from tqdm import trange
import re
from time import time, sleep


def parse_args():
    """# Set hyper-parameters """

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.00025,
                        help=' initial learning rate')

    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='discount factor, also known as gamma ')

    parser.add_argument('--epochs', type=int, default=20,
                        help=' number of epochs')

    parser.add_argument('--steps_per_epoch', type=int, default=5000,
                        help=' number of steps per epochs')

    parser.add_argument('--replay_memory_size', type=int, default=25000,
                        help=' length of replay memory buffer')

    parser.add_argument('--batch_size', type=int, default=64,
                        help=' length of sampling from replay memory')

    parser.add_argument('--tests_per_epoch', type=int, default=5,
                        help=' number of test episodes per epoch')

    parser.add_argument('--frame_repeat', type=int, default=8,
                        help='number of frames last an action')

    parser.add_argument('--resolution', type=tuple, default=(84, 84),
                        help='dimensions of CNN input. Warning value must be a tuple of (width,height)')

    parser.add_argument('--episodes_to_watch', type=int, default=15,
                        help=' number of episodes to watch while evaluating the agent ')

    parser.add_argument('--scenario', type=str, default="scenario/take_cover",
                        help=' vizdoom scenario and cfg location. Warning, do not precise type, and .wad & .cfg must be in the same dir')

    parser.add_argument('--model_savedfile', type=str, default="",
                        help=' Saved model location if None, it will take the latest')

    parser.add_argument('--save_model', type=bool, default=True,
                        help=' boolean to save the model while training')

    parser.add_argument('--load_model', type=bool, default=False,
                        help='  if you want to load an existing model.')

    parser.add_argument('--skip_learning', type=bool, default=False,
                        help=' if you want to skip the learning. Only usable if a model is loaded !')

    parser.add_argument('--end_eps', type=float, default=0.1,
                        help=' the lowest epsilon can be')

    parser.add_argument('--log_freq', type=int, default=1,
                        help=' How often the logging shall be done')

    parser.add_argument('--wandering', type=int, default=5000,
                        help=' First memory initialization')

    return parser.parse_args()


def initialize_vizdoom(config_file_path):
    """# Initialize vizdoom environment with some basic parameters"""

    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_doom_scenario_path(flags.scenario + ".wad")
    game.set_doom_map("map01")
    game.set_screen_format(ScreenFormat.GRAY8)
    print("Doom initialized.")
    return game


def init():
    """# Instantiate environment  & get last trained model"""

    # Create Doom instance
    game = initialize_vizdoom(flags.scenario + ".cfg")

    # load latest model
    if flags.model_savedfile == "" and flags.load_model:
        test = getfile(os.path.join(os.getcwd() + "/logs/"), [".pth"])
        assert (len(test) != 0), "No model available or found please select one or set load model to False"
        dates = []

        for path in test:
            dates.append(re.findall("\/logs/([0-9._\-]+)", path)[0])

        flags.model_savedfile = test[dates.index(max(dates))]

    return game


def display_results(score):
    """# Display average, min, max, and  variance of given score"""

    scores = np.array(score)

    print("Results: mean: %.1f +/- %.1f," % (scores.mean(), scores.std()),
          "min: %.1f," % scores.min(), "max: %.1f," % scores.max())


def train(agent):
    """# Training loop of agent learning """

    game.init()
    time_start = time()
    game.new_episode()

    for _ in trange(flags.wandering):
        s1 = preprocess(game.get_state().screen_buffer, flags.resolution)
        s1 = s1.reshape([1, 1, flags.resolution[0], flags.resolution[1]])
        a = randint(0, len(agent.action_map) - 1)
        reward = game.make_action(agent.action_map[a], flags.frame_repeat)
        reward = norm(reward)

        isterminal = game.is_episode_finished()
        s2 = preprocess(game.get_state().screen_buffer, flags.resolution).reshape(
            [1, 1, flags.resolution[0], flags.resolution[1]]) if not isterminal else None

        agent.add_mem(s1, a, s2, isterminal, reward)

        if game.is_episode_finished():
            game.new_episode()

    for epoch in range(flags.epochs):
        print("\nEpoch %d\n-------" % (epoch + 1))
        train_episodes_finished = 0
        train_scores = []

        print("Training...")
        game.new_episode()
        agent.model.reset_hidden()
        for learning_step in trange(flags.steps_per_epoch, leave=False):
            s1 = preprocess(game.get_state().screen_buffer, flags.resolution)
            s1 = s1.reshape([1, 1, flags.resolution[0], flags.resolution[1]])
            # With probability eps make a random action.
            eps = exploration_rate(epoch, learning_step)
            if random() <= eps:
                a = randint(0, len(agent.action_map) - 1)
            else:
                # Choose the best action according to the network.

                a = agent.get_best_action(s1)

            reward = game.make_action(agent.action_map[a], flags.frame_repeat)
            reward = norm(reward)

            isterminal = game.is_episode_finished()
            s2 = preprocess(game.get_state().screen_buffer, flags.resolution).reshape(
                [1, 1, flags.resolution[0], flags.resolution[1]]) if not isterminal else None

            agent.add_mem_and_learn(s1, a, s2, isterminal, reward)

            if game.is_episode_finished():
                score = game.get_total_reward()
                train_scores.append(score)

                game.new_episode()
                agent.model.reset_hidden()
                train_episodes_finished += 1

        print("Epsilon : %.5f" % eps)
        print("%d training episodes played." % train_episodes_finished)

        display_results(train_scores)

        print("\nTesting...")

        test_scores = []
        d = 0
        f = 0
        for ep in trange(flags.tests_per_epoch, leave=False):
            if epoch % flags.log_freq == 0:
                log.make_epoch_dir(epoch)
                save_model_weight(epoch, agent)

            game.new_episode()
            agent.model.reset_hidden()
            w = 0

            while not game.is_episode_finished():

                s1 = preprocess(game.get_state().screen_buffer, flags.resolution)
                s1 = s1.reshape([1, 1, flags.resolution[0], flags.resolution[1]])

                best_action_index = agent.get_best_action(s1)
                if best_action_index == 0:
                    d += 1
                else:
                    f += 1

                game.make_action(agent.action_map[best_action_index], flags.frame_repeat)

                if game.get_state() is not None:
                    save_model_features(epoch, w, agent, ep)

                    if epoch % flags.log_freq == 0:
                        save_inputs(game.get_state().screen_buffer,
                                    s1.reshape(flags.resolution[0], flags.resolution[1]),
                                    ep, epoch, w)
                w += 1
            r = game.get_total_reward()
            log.save_reward(epoch, ep, r)
            test_scores.append(r)

        print("-------------------")
        print("action 0:", d, " VS action 1:", f)
        print("-------------------")

        display_results(test_scores)

        if flags.save_model:
            agent.save_model(log.models_dir + "/model.pth")
        print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()


def evaluation(agent):
    """# Run trained agent and gather model information to display """

    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(flags.episodes_to_watch):
        game.new_episode()
        agent.model.reset_hidden()

        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, flags.resolution)
            state = state.reshape([1, 1, flags.resolution[0], flags.resolution[1]])
            best_action_index = agent.get_best_action(state)
            game.set_action(agent.action_map[best_action_index])
            for _ in range(flags.frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)


def save_model_weight(epoch, agent):
    temp = agent.model.get_info()

    log.make_dir("weights")

    for w in range(2):
        for y in range(temp[w].shape[0]):
            log.save_weight_gray(temp[w][y][0],
                                 "weight_layer" + str(w) + "_" + "unit" + str(y) + "_" + "epoch" + str(epoch) + ".jpg")


def save_model_features(epoch, i, agent, episode):
    temp = agent.model.get_info()
    log.make_episode(episode)

    log.make_instance_dir("features")
    log.make_instance_dir("hidden")
    log.make_instance_dir("acts")

    for w in range(2, len(temp) - 3):
        for y in range(temp[w].shape[1]):
            log.save_feature_gray(temp[w][0][y], "feature_layer" + str(w - 2) + "_" + "unit" + str(y) + "_epoch" + str(
                epoch) + "_act" + str(i) + ".jpg")
    h = "["
    for elem in temp[(len(temp) - 3)][0, 0]:
        h += str(elem) + ","
    h = h[:-1] + "]"

    log.save_vector(h, "hidden", "hidden_epoch" + str(epoch) + "_act" + str(i) + ".json")

    q = "["
    for elem in temp[(len(temp) - 1)][0, 0]:
        q += str(elem) + ","
    q = q[:-1] + "]"
    log.save_vector(q, "acts", "acts_epoch" + str(epoch) + "_act" + str(i) + ".json")


def save_inputs(img1, img2, episode, epoch, act):
    name1 = "input" + str(act) + "_epoch" + str(epoch) + "_episode" + str(episode) + ".jpg"
    name2 = "preprocess_input" + str(act) + "_epoch" + str(epoch) + "_episode" + str(episode) + ".jpg"

    log.make_instance_dir("images")
    log.make_instance_dir("preimages")

    log.save_input(img1, name1, 0)
    log.save_input(img2, name2, 1)


def exploration_rate(epoch, i):
    """# Define and apply epsilon decay over time"""

    if epoch < flags.epochs - (round(flags.epochs * 0.2)) > 0:
        return math.exp(-1.5 * float(i * epoch) / float(flags.steps_per_epoch * round(flags.epochs * 0.8)))
    else:
        return flags.end_eps


if __name__ == '__main__':

    flags = parse_args()

    game = init()

    log = Log()

    agent = RdqnAgent(flags)

    if not flags.skip_learning:
        print("Starting the training!")
        train(agent)
        print("======================================")
        input("Training finished. It's time to watch ! press any key to watch the result")
    evaluation(agent)
