import numpy as np
from stable_baselines3 import A2C
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.environment import SideCondition
from poke_env.player import Gen9EnvSinglePlayer, RandomPlayer,SimpleHeuristicsPlayer

MAX_TURNS = 255
# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen9EnvSinglePlayer):
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

        # hp
        remaining_health = np.ones(6)
        opponent_remaining_health = np.ones(6)
        # stats
        stats = np.zeros(7)
        opponent_stats = np.zeros(7)
        # status
        status = np.zeros(6)
        opponent_status = [mon.status for mon in battle.opponent_team.values()]
        for i, mon in enumerate(battle.team.values()):
            remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    stats[j] = b
        for i, mon in enumerate(battle.opponent_team.values()):
            opponent_remaining_health[i] = mon.current_hp_fraction
            if mon.status is not None:
                opponent_status[i] = mon.status.value
            if mon.active:
                for j,b in enumerate(mon.boosts.values()):
                    opponent_stats[j] = b

        # tera
        tera = bool(battle.can_tera)
        opponent_tera = bool(battle._opponent_can_terrastallize)

        # hazards
        hazarddex = { # relevant side conditions to track (to reduce this dimension by half and not have variable sized input)
            SideCondition.UNKNOWN:0,
            SideCondition.AURORA_VEIL:1,
            SideCondition.LIGHT_SCREEN:2,
            SideCondition.MIST:3,
            SideCondition.REFLECT:4,
            SideCondition.SAFEGUARD:5,
            SideCondition.SPIKES:6,
            SideCondition.STEALTH_ROCK:7,
            SideCondition.STICKY_WEB:8,
            SideCondition.TAILWIND:9,
            SideCondition.TOXIC_SPIKES:10,
        }
        hazards = np.zeros(11)
        for cond,val in battle.side_conditions.items():
            if cond in hazarddex:
                hazards[hazarddex[cond]] = val
        opponent_hazards = np.zeros(11)
        for cond,val in battle.opponent_side_conditions.items():
            if cond in hazarddex:
                opponent_hazards[hazarddex[cond]] = val

        # terrain
        fields = np.zeros(13)
        for f,val in battle.fields.items():
            fields[f.value-1] = val

        # weather
        weather = np.zeros(9)
        for w,val in battle.weather.items():
            weather[w.value-1] = val

        # turn
        turn = battle.turn

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power, # 4* -1,3
                moves_dmg_multiplier, # 4* 0,4
                [remaining_mon_team, remaining_mon_opponent], # 1*1* 0,1 0,1
                remaining_health, # 6* 0,1
                opponent_remaining_health, # 6* 0,1
                [tera, opponent_tera], # 1*1* 0,1 0,1
                stats, # 7* -6,6
                opponent_stats, # 7* -6,6
                hazards, # 11* 0,MAX_TURNS
                opponent_hazards, # 11* 0,MAX_TURNS
                fields, # 13* 0,MAX_TURNS
                weather, # 9* 0,MAX_TURNS
                [turn], # 1* 0,MAX_TURNS
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )+25/current_state.turn
    
    def describe_embedding(self):
        low = np.concatenate([[-1]*4, [0]*4, [0]*2, [0]*6, [0]*6, [0]*2, [-6]*7, [-6]*7, [0]*11, [0]*11, [0]*13, [0]*9, [0]])
        high = np.concatenate([[3]*4, [4]*4, [1]*2, [1]*6, [1]*6, [1]*2, [6]*7, [6]*7, [MAX_TURNS]*11, [MAX_TURNS]*11, [MAX_TURNS]*13, [MAX_TURNS]*9, [MAX_TURNS]])
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move,_= model.predict(obs, deterministic=True)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

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

class A2Cplayer(RandomPlayer):

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()
    return action

NB_EVALUATION_EPISODES = 100

np.random.seed(0)


model_store = {}

# This is the function that will be used to train the a2c
def a2c_training(player, nb_steps):
    model = A2C("MlpPolicy", player, verbose=1)
    model.learn(total_timesteps=10_000)
    model_store[player] = model
    


def a2c_evaluation(player, nb_episodes):
    # Reset battle statistics
    model = model_store[player]
    player.reset_battles()
    model.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )


NB_TRAINING_STEPS = 10_000
TEST_EPISODES = 100
GEN_9_DATA = GenData.from_gen(9)

if __name__ == "__main__":
    opponent = RandomPlayer()
    env_player = SimpleRLPlayer(opponent=opponent)
    second_opponent = MaxDamagePlayer()
    third_op=SimpleHeuristicsPlayer()
    a2c_training(env_player,NB_TRAINING_STEPS)
    a2c_evaluation(env_player,TEST_EPISODES)
    model_store[env_player].env._opponent=second_opponent
    model_store[env_player].learn(NB_TRAINING_STEPS)
    a2c_training(env_player,NB_TRAINING_STEPS)
    a2c_evaluation(env_player,TEST_EPISODES)
    model_store[env_player].env._opponent=third_op
    model_store[env_player].learn(NB_TRAINING_STEPS)
    a2c_training(env_player,NB_TRAINING_STEPS)
    a2c_evaluation(env_player,TEST_EPISODES)
    obs, reward, done, _, info = env_player.step(0)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

    finished_episodes = 0

    env_player.reset_battles()
    obs, _ = env_player.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            if finished_episodes >= TEST_EPISODES:
                break
            obs, _ = env_player.reset()

    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)

    finished_episodes = 0

    env_player.reset_env(restart=True, opponent=MaxDamagePlayer())

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_player.step(action)

        if done:
            finished_episodes += 1
            obs, _ = env_player.reset()
            if finished_episodes >= TEST_EPISODES:
                break

        player = env_player.agent
        #SimpleRLPlayer(player_configuration=AccountConfiguration("Agent_ego", "RLclass") server_configuration=ShowdownServerConfiguration,)
        # Sending challenges to 'your_username'
        #await player.send_challenges("your_username", n_challenges=1)

        # Accepting one challenge from any user
        #await player.accept_challenges(None, 1)

        # Accepting three challenges from 'your_username'
        #await player.accept_challenges('your_username', 3)

        # Playing 5 games on the ladder
        
        await player.ladder(5)
    async def main():
        # We create a random player
        player = SimpleRLPlayer(
            player_configuration=AccountConfiguration("Agent_ego", "RLclass")
            server_configuration=ShowdownServerConfiguration,
        )
        # Sending challenges to 'your_username'
        #await player.send_challenges("your_username", n_challenges=1)

        # Accepting one challenge from any user
        #await player.accept_challenges(None, 1)

        # Accepting three challenges from 'your_username'
        #await player.accept_challenges('your_username', 3)

        # Playing 5 games on the ladder
        
        await player.ladder(5)


        # Print the rating of the player and its opponent after each battle
        for battle in player.battles.values():
            print(battle.rating, battle.opponent_rating)

    asyncio.get_event_loop().run_until_complete(main())
    print("Won", env_player.n_won_battles, "battles against", env_player._opponent)