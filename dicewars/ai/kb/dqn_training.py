import random
import numpy as np
import pandas as pd
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import os
import pickle
import json
from dicewars.ai.kb.move_selection import get_transfer_from_endangered, get_transfer_to_border
from dicewars.ai.utils import possible_attacks, probability_of_holding_area, probability_of_successful_attack
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board
from colorama import Fore, Style

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def define_parameters():
    params = dict()
    # Neural Network
    cnt = 100
    params['epsilon_decay_linear'] = 1/cnt
    # params['learning_rate'] = 0.00013629
    params['learning_rate'] = 0.003
    params['first_layer_size'] = 200        # neurons in the first layer
    params['second_layer_size'] = 60        # neurons in the second layer
    params['third_layer_size'] = 30         # neurons in the third layer
    params['episodes'] = cnt   
    params['memory_size'] = 2000
    params['batch_size'] = 300
    # Settings
    params['weights_path'] = os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/weights/weights-final.h5')
    TrainAndNotLoad = True
    params['train'] = TrainAndNotLoad
    params['load_weights'] = not TrainAndNotLoad
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params

class AI(torch.nn.Module):
    def __init__(self, player_name, board, players_order, max_transfers):
        super().__init__()
        print("New AI object was created...")
        self.player_name = player_name
        self.params = define_parameters()
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.learning_rate = self.params['learning_rate']        
        self.epsilon = 1
        self.first_layer = self.params['first_layer_size']
        self.second_layer = self.params['second_layer_size']
        self.third_layer = self.params['third_layer_size']
        self.memory = collections.deque(maxlen=self.params['memory_size'])
        self.weights = self.params['weights_path']
        self.load_weights = self.params['load_weights']
        self.counter_games = 0
        self.performed_attacks = 0
        self.player_areas_old = 1
        self.max_transfers = max_transfers
        self.reserve_for_evacuation = 2 if 2 < max_transfers else 0
        
        self.final_move_old = None
        self.final_moves_cache = []

        self.bad_prediction = False

        self.fresh_start = True
        self.num_of_turns = 0
        self.num_of_model_predictions = 0
        self.num_of_bad_predictions = 0

        # loss values
        self.loss_vals = []
        self.epoch_loss= []

        self.config, self.memory_snap = self.load_ai_state()
        self.network()

        if self.config:
            print("Loading state config from previous game")
            self.epsilon = self.config['epsilon']
            self.counter_games = self.config['counter_games']
            self.loss_vals = self.config['loss_vals']
            
            print(f"""
                    Previous state loaded, new values:
                    Epsilon: {self.epsilon},
                    Game counter: {self.counter_games}
                    """)
        
        if self.memory_snap:
            print("Loading memory from previous game")
            self.memory = self.memory_snap
        
        if self.params['train']:
            print("Episodes: {}/{}".format(self.counter_games, self.params['episodes']))

    def __del__(self):
        print("Total turns: {}, Model made {} predictions, {} of them were bad.".format(self.num_of_turns, self.num_of_model_predictions, self.num_of_bad_predictions))
        if self.params['train']:
            model_weights = self.state_dict()
            torch.save(model_weights, self.params["weights_path"])

        if self.params['episodes'] == self.counter_games:
            print(f"{Fore.GREEN}Deleting state files... {Style.RESET_ALL}")
            os.remove(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_STATE.pickle'))
            

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        
        self.epsilon = 0.05 if not self.params['train'] else 1 - (self.counter_games * self.params['epsilon_decay_linear'])
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.params['learning_rate'])
        state_new, attacks = self.get_state(board, nb_moves_this_turn)

        #
        #   Transfer commands
        #

        # move dices to borders
        if nb_transfers_this_turn < self.max_transfers:
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])

        # Evacuation plan
        if len(attacks) == 0 or self.performed_attacks >= 10:
            if nb_transfers_this_turn < self.max_transfers:
                transfer = get_transfer_from_endangered(board, self.player_name)
                if transfer:
                    return TransferCommand(transfer[0], transfer[1])
                else:
                    transfer = get_transfer_to_border(board, self.player_name)
                    if transfer:
                        return TransferCommand(transfer[0], transfer[1])

            self.performed_attacks = 0
            self.save_ai_state()
            print(f"Predicted indexes: {self.final_moves_cache}")
            self.final_moves_cache.clear()
            return EndTurnCommand()

        # retrain on entire memory if last epoch has ended (this is start of the new game)
        if self.num_of_turns == 0 and self.params['train']:
            self.player_areas_old = len(board.get_player_areas(self.player_name))
            if len(self.memory) > 0:
                self.replay_new(self.memory, self.params['batch_size'])

        # get results of previous round and train network (if its not the first round)
        if self.num_of_turns != 0 and self.params['train']:
            self.player_areas_current = len(board.get_player_areas(self.player_name))
            reward = self.set_reward(self.player_areas_current, self.player_areas_old)
            print(f"({self.player_areas_old}) --> ({self.player_areas_current})  =  {reward}")
            
            # train short memory base on the new action and state
            # self.train_short_memory(self.state_old, self.final_move_old, reward, state_new)
            # store the new data into a long term memory
            self.remember(self.state_old, self.final_move_old, reward, state_new)

        # remember current state for next turn
        self.state_old = state_new
        
        NN_predicted = False
        self.bad_prediction = False
        final_move_index = -1
        if random.uniform(0, 1) < self.epsilon:
            random_index = random.randint(0, len(attacks) - 1 if len(attacks) - 1 <= 5 else 5)
            final_move = np.eye(6)[random_index]
            self.final_move_old = final_move
            final_move_index = random_index
        else:
            # predict action based on the old state
            with torch.no_grad():
                NN_predicted = True
                state_old_tensor = torch.tensor(state_new.reshape((1, 42)), dtype=torch.float32).to(DEVICE)
                prediction = self(state_old_tensor)
                final_move_index = np.argmax(prediction.detach().cpu().numpy()[0])
                final_move = np.eye(6)[final_move_index]
                self.final_move_old = final_move
                self.num_of_model_predictions += 1
                self.final_moves_cache.append(final_move_index)
                
                # print("NN Output: {}, attacks len: {}".format(prediction.detach().cpu().numpy()[0], len(attacks)))
                if len(attacks) <= final_move_index:
                    
                    max_prob = 0
                    max_prob_index = 0
                    for i, attack in enumerate(attacks):
                        prob = probability_of_successful_attack(board, attack[0].get_name(), attack[1].get_name())
                        if max_prob < prob:
                            max_prob = prob
                            max_prob_index = i

                    final_move = np.eye(6)[max_prob_index]
                    self.final_move_old = final_move
                    final_move_index = max_prob_index
                    self.bad_prediction = True
                    self.num_of_bad_predictions += 1

                

                    
        src_target = attacks[final_move_index]
        # print("[{}] Attack: {} ({}) -> {} ({}), prob. of success: {} {} (index {}, attacks len: {})".format("NN" if NN_predicted else "RD", src_target[0].get_name(), src_target[0].get_dice(), src_target[1].get_name(), src_target[1].get_dice(), probability_of_successful_attack(board, src_target[0].get_name(), src_target[1].get_name()), "[Bad prediction]" if self.bad_prediction else "", final_move_index, len(attacks)))
        self.player_areas_old = len(board.get_player_areas(self.player_name))
        self.num_of_turns += 1
        self.performed_attacks += 1
        return BattleCommand(src_target[0].get_name(), src_target[1].get_name())  
    
    def save_ai_state(self):
        
        # print(f"Saving state, mem len {len(self.memory)}")
        # print("AI is being destroyed, saving current model state...")
        if self.fresh_start:
            self.fresh_start = False
            self.counter_games += 1
       
       # Pickle
        with open(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_STATE.pickle'), 'wb') as config_dictionary_file:
            state = {
                "epsilon": self.epsilon,
                "counter_games": self.counter_games,
                "loss_vals": self.loss_vals
            }
            pickle.dump(state, config_dictionary_file)

        with open(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_MEMORY.pickle'), 'wb') as memory_file:
            try:
                pickle.dump(self.memory, memory_file)
            except Exception as e:
                print(" --- Saving memory error --- ")
                print(e)


    def load_ai_state(self):
        config = None
        memory = None
        if os.path.exists(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_STATE.pickle')):
            with open(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_STATE.pickle'), 'rb') as config_dictionary_file:
                print("reading config")
                config = pickle.load(config_dictionary_file)

        if os.path.exists(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_MEMORY.pickle')):
            with open(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_MEMORY.pickle'), 'rb') as memory_file:
                try:
                    memory = pickle.load(memory_file)
                except Exception as e:
                    print("Memory corrupt")
        return config, memory

    def network(self):
        # Layers
        self.f1 = nn.Linear(42, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 6)

        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("Weights were loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, board: Board, moves_this_turn):
        state = []
        attacks = [a for a in possible_attacks(board, self.player_name) if a[0].get_dice() > (a[1].get_dice() - (0 if moves_this_turn else 1))]
        attacks_sorted = sorted(attacks, key=lambda x: probability_of_successful_attack(board, x[0].get_name(), x[1].get_name()), reverse=True)
        if attacks:
            for i, attack in enumerate(attacks_sorted): 
                if i == 6: break
                src: Area = attack[0]
                dst: Area = attack[1]
                prob_of_success = probability_of_successful_attack(board, src.get_name(), dst.get_name())
                prob_of_hodl = probability_of_holding_area(board, src.get_name(), src.get_dice(), self.player_name)
                areas_around: list[int] = src.get_adjacent_areas_names()
                
                max_dice_around_target = 0
                sum_dice_around_target = 0
                for area_name in dst.get_adjacent_areas_names():
                    target_area: Area = board.get_area(area_name)
                    if target_area.get_owner_name() != self.player_name:
                        if max_dice_around_target < target_area.get_dice():
                            max_dice_around_target = target_area.get_dice()
                            sum_dice_around_target += target_area.get_dice()

                num_of_enemies_around = 0
                for area in areas_around:
                    if board.get_area(area).get_owner_name() != self.player_name: 
                        num_of_enemies_around += 1
                
                state.append(float(prob_of_success))
                state.append(float(prob_of_hodl))
                state.append(float(num_of_enemies_around))
                state.append(float(max_dice_around_target))
                state.append(float(sum_dice_around_target))
                state.append(float(src.get_dice()))
                state.append(float(dst.get_dice()))
                
        for i in range(len(state), 42, 7):
            state.append(0.00)
            state.append(1.00)
            state.append(1.00)
            state.append(8.00)
            state.append(45.00)
            state.append(2.00)
            state.append(8.00)

        return np.asarray(state), attacks_sorted

    def set_reward(self, num_of_areas: int, last_num_of_areas: int):
        
        self.reward = 0
        if self.bad_prediction:
            self.reward = -10
        elif num_of_areas < last_num_of_areas:
            self.reward = -5
        elif num_of_areas > last_num_of_areas:
            self.reward = 10

        # print("Reward: {}".format(self.reward))
        # print("current areas: {}, old areas: {}".format(num_of_areas, last_num_of_areas))
        return self.reward

    def remember(self, state, action, reward, next_state):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        
        # if mem len is lesser than batch_size, there is no point in learning
        if batch_size > len(memory):
            return

        print("Starting replaying memory, mem len: {}".format(len(memory)))
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory

        for state, action, reward, next_state in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            # target = reward

            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            
            q_eval = self.forward(state_tensor)
            # q_eval = self.forward(state_tensor)[0][np.argmax(action)]
            q_next = self.forward(next_state_tensor)
            q_target = reward + self.gamma * torch.max(q_next[0])

        
            """
            loss = F.mse_loss(q_eval, q_target).to(DEVICE)
            loss.backward()
            self.optimizer.step()
            self.epoch_loss.append(loss.item())
            """

            target_f = q_eval.clone()
            target_f[0][np.argmax(action)] = q_target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(q_eval, target_f)
            loss.backward()
            self.epoch_loss.append(loss.item())
            self.optimizer.step()    



        self.loss_vals.append(sum(self.epoch_loss)/len(self.epoch_loss))
        print("Loss value {}".format(self.loss_vals[-1]))
        print("Replay finished")       

    def train_short_memory(self, state, action, reward, next_state):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 42)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 42)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        # print("Loss value {}".format(loss.item()))
        self.optimizer.step()
