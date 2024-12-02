import numpy as np
import pandas as pd
import asyncio
from stable_baselines3 import A2C,DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.spaces import Box

from poke_env.data import GenData
from poke_env.environment import SideCondition
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer, SimpleHeuristicsPlayer, Player
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

import klefki
#import tqdm

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, env_player, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.env_player = env_player

    def _on_step(self) -> bool:
        """This method is required but not used for reset logic."""
        return True  # Allow training to continue.

    def on_training_end(self) -> None:
        """Called when the training phase ends, ensuring cleanup."""
        print("Training phase completed. Resetting environment.")
        self.env_player.reset_env()  # Clean up battles safely

MAX_TURNS = 255
# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class Agent(Gen9EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:

        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float64),
            np.array(high, dtype=np.float64),
            dtype=np.float64,
        )

class OnlineAgent(Player):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_9_DATA.type_chart
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:

        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float64),
            np.array(high, dtype=np.float64),
            dtype=np.float64,
        )
    
    def action_to_move(self, action, battle):
        if action == -1:
            return ForfeitBattleOrder()
        elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 8], mega=True
            )
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 12], dynamax=True
            )
        elif (
            battle.can_tera
            and 0 <= action - 16 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(
                battle.available_moves[action - 16], terastallize=True
            )
        elif 0 <= action - 20 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 20])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle):
        obs = self.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.action_to_move(action)

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


np.random.seed(0)
TEST_EPISODES=100
def evaluate_policy(model):
    opponents = [
        RandomPlayer(battle_format='gen9randombattle'),
        MaxDamagePlayer(battle_format='gen9randombattle'),
        SimpleHeuristicsPlayer(battle_format='gen9randombattle'),
    ]
    results = {}

    for opponent in opponents:
        print(f"Evaluating against {type(opponent).__name__}")
        env_player = Agent(opponent=opponent)
        env_player.reset_battles()
        model.set_env(env_player)
        
        finished_episodes = 0
        total_rewards = 0
        
        for _ in tqdm.tqdm(range(TEST_EPISODES)):
            obs, _ = env_player.reset()  # Reset the environment at the start of each episode
            done = False
            
            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, _ = env_player.step(action)
                    total_rewards += reward
                except RuntimeError as e:
                    print(f"Error: {e}")
                    done = True  # Force end of episode if a battle-related error occurs
            
            finished_episodes += 1
            # print(f"Episode {finished_episodes} completed")
        
        # Record the results after finishing all episodes with the current opponent
        results[type(opponent).__name__] = env_player.n_won_battles
        print(f"Won {env_player.n_won_battles} battles against {type(opponent).__name__}")
    
    # Display evaluation results
    print("Evaluation Results:")
    for opponent_name, wins in results.items():
        print(f"{opponent_name}: {wins}/{TEST_EPISODES} wins")
    
    return results

def a2c_evaluation(csv=True):
    results_list = []
    
    # List of timesteps to evaluate
    timesteps = [3]
    
    for ts in timesteps:
        print(f"Evaluating model trained for {ts} timesteps...")
        
        # Load the model for the specific timestep
        model = A2C.load(f'a2c_model_{ts}')
        
        # Evaluate the policy against the opponents
        results = evaluate_policy(model)
        
        # Store results in a list
        if results:
            for opponent, victories in results.items():
                results_list.append({"timestep": ts, "opponent": opponent, "wins": victories})
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results_list)
        
    if csv:
        # Save DataFrame to a CSV file
        results_df.to_csv('a2c_evaluation_results.csv', index=False)
        print("Results saved to 'a2c_evaluation_results.csv'.")

    # Display the results for quick verification
    print("Results a2c")
    print(results_df)

def dqn_evaluation(csv=True):
    results_list = []
    
    # List of timesteps to evaluate
    timesteps = [3]
    
    for ts in timesteps:
        print(f"Evaluating model trained for {ts} timesteps...")
        
        # Load the model for the specific timestep
        model = DQN.load(f'dqn_model_{ts}')
        
        # Evaluate the policy against the opponents
        results = evaluate_policy(model)
        
        # Store results in a list
        if results:
            for opponent, victories in results.items():
                results_list.append({"timestep": ts, "opponent": opponent, "wins": victories})
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results_list)
        
    if csv:
        # Save DataFrame to a CSV file
        results_df.to_csv('dqn_evaluation_results.csv', index=False)
        print("Results saved to 'dqn_evaluation_results.csv'.")

    # Display the results for quick verification
    print("Results dqn")
    print(results_df)

# play online
async def a2cladder():
    print('Entering the ladder...')
    # We create a random player
    model = model_store["a2c"]
    player = OnlineAgent(
        account_configuration=AccountConfiguration(username=klefki.a2cuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
        model=model,
        start_timer_on_battle_start=True,
    )
    print('It\'s a bad day to be a human pokemon trainer')
    await player.ladder(LADDER_EPISODES)
    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
         print(battle.rating, battle.opponent_rating)

async def dqnladder():
    # We create a random player
    model = model_store["dqn"]
    player = Agent(
        player_configuration=AccountConfiguration(username=klefki.dqnuser, password=klefki.password),
        server_configuration=ShowdownServerConfiguration,
    )

    model.env = player
    await model.env.ladder(LADDER_EPISODES)

    # Print the rating of the player and its opponent after each battle
    for battle in player.battles.values():
        print(battle.rating, battle.opponent_rating)


if __name__ == "__main__":
    #a2c_training(2_000_000)
    #a2c_training(1_000_000)
    #dqn_training(500000)

    # a2c_training(5000)
    # dqn_training(5000)

    a2c_evaluation(csv=False)
    dqn_evaluation(csv=False)

    # asyncio.get_event_loop().run_until_complete(a2cladder())
