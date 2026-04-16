# test.py
import cv2

from params import TEST_TOWN, TEST_NUM_EPISODES
from utils.test_utils import build_models, run_evaluation, print_report
from env.carla_wrapper import CarlaEnv


def main():
    print(f"\n>>> Starting Evaluation on {TEST_TOWN} <<<")

    env = CarlaEnv(town=TEST_TOWN)
    encoder, rssm, actor, decoder = build_models()

    try:
        results = run_evaluation(env, encoder, rssm, actor, decoder,
                                 num_episodes=TEST_NUM_EPISODES)
        print_report({TEST_TOWN: results})
    finally:
        cv2.destroyAllWindows()
        env._cleanup()


if __name__ == "__main__":
    main()