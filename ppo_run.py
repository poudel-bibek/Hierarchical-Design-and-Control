import argparse
from sim_run import CraverRoadEnv

def main(args):

    env = CraverRoadEnv(args)
    observation, info = env.reset()

    for _ in range(args.max_steps):
        action = env.action_space.sample()
        print(f"\nStep {env.step_count}: Taking action {action}")
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUMO traffic simulation with demand scaling.')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI (default: False)')
    parser.add_argument('--step-length', type=float, default=1.0, help='Simulation step length (default: 1.0)')
    parser.add_argument('--max_steps', type=int, default=2500, help='Maximum number of steps in one episode (default: 2000)')
    parser.add_argument('--auto_start', action='store_true', help='Automatically start the simulation (default: False)')

    parser.add_argument('--input_trips', type=str, default='./original_vehtrips.xml', help='Original Input trips file')
    parser.add_argument('--output_trips', type=str, default='./scaled_vehtrips.xml', help='Output trips file')
    parser.add_argument('--scale_demand', type=bool , default=True, help='Scale demand before starting the simulation')
    parser.add_argument('--scale_factor', type=float, default=3.0, help='Demand scaling factor (default: 1.0)')
    args = parser.parse_args()

    main(args)