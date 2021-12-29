import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
from dicewars.ai.kb.move_selection import get_transfer_from_endangered
from dicewars.ai.utils import possible_attacks, probability_of_holding_area, probability_of_successful_attack
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board
from colorama import Fore, Style


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = False

def define_parameters():
    params = dict()

    # Neural Network
    cnt = 200                                   # num of episodes
    params['epsilon_decay_linear'] = 1/200
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200            # neurons in the first layer
    params['second_layer_size'] = 60            # neurons in the second layer
    params['third_layer_size'] = 30             # neurons in the third layer
    params['episodes'] = cnt   
    params['memory_size'] = 2000
    params['batch_size'] = 300

    # Settings
    params['weights_path'] = os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/weights/weights-final.h5')
    TrainAndNotLoad = True
    params['train'] = TrainAndNotLoad
    params['load_weights'] = not TrainAndNotLoad
    return params

# debug dprint
def dprint(msg):
    if DEBUG: dprint(msg)

class AI(torch.nn.Module):

    def __init__(self, player_name, board, players_order, max_transfers):
        super().__init__()
        dprint("\n--------------------------------\nNew AI object was created...\n--------------------------------")
        self.player_name = player_name
        self.players_order = players_order
        self.params = define_parameters()
        self.reward = 0
        self.gamma = 0.9
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
        self.final_moves_all = []

        self.bad_prediction = False

        self.fresh_start = True
        self.num_of_turns = 0
        self.num_of_model_predictions = 0
        self.num_of_bad_predictions = 0

        # loss values
        self.loss_vals = [100]
        self.epoch_loss= []

        self.config, self.memory_snap = self.load_ai_state()
        self.network()

        if self.config:
            dprint("Loading state config from previous game")
            self.epsilon = self.config['epsilon']
            self.counter_games = self.config['counter_games']
            self.loss_vals = self.config['loss_vals']
            
            dprint(
            f"""
            Previous state loaded, new values:
            Epsilon: {self.epsilon},
            Game counter: {self.counter_games},
            Loss value: {self.loss_vals[-1]}
            """)
        
        if self.memory_snap:
            dprint("Loading memory from previous game")
            self.memory = self.memory_snap
        
        if self.params['train']:
            dprint("Episodes: {}/{}".format(self.counter_games, self.params['episodes']))

    def __del__(self):
        dprint("Total turns: {}, Model made {} predictions, {} of them were bad.".format(self.num_of_turns, self.num_of_model_predictions, self.num_of_bad_predictions))
        dprint(f"Predicted indexes: {self.final_moves_all}")
        if self.params['train']:
            model_weights = self.state_dict()
            torch.save(model_weights, self.params["weights_path"])

        if self.params['episodes'] == self.counter_games:
            dprint(f"{Fore.GREEN}Deleting state files... {Style.RESET_ALL}")
            os.remove(os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/pickles/DQN_STATE.pickle'))
            

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        
        self.epsilon = 0.05 if not self.params['train'] else 1 - (self.counter_games * self.params['epsilon_decay_linear'])
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=self.params['learning_rate'])
        state_new, attacks = self.get_state(board, nb_moves_this_turn)

        #
        #   Transfer commands
        #
        
        # move dices to borders
        if nb_transfers_this_turn < self.max_transfers:
            transfer = self.get_transfer_to_border_custom(board, self.player_name, 4)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])

        # Evacuation plan
        if len(attacks) == 0 or self.performed_attacks >= 10:
            if nb_transfers_this_turn < self.max_transfers:
                transfer = get_transfer_from_endangered(board, self.player_name)
                if transfer:
                    return TransferCommand(transfer[0], transfer[1])
                else:
                    # if no evacuation plan is needed, transfer all you can to borders (even areas with 2 can now be transfered from) 
                    transfer = self.get_transfer_to_border_custom(board, self.player_name, 2)
                    if transfer:
                        return TransferCommand(transfer[0], transfer[1])

            self.performed_attacks = 0
            self.save_ai_state()
            self.final_moves_all.append(self.final_moves_cache.copy())
            self.final_moves_cache.clear()
            return EndTurnCommand()

        # retrain on entire memory if last epoch has ended (this is start of the new game)
        if self.num_of_turns == 0 and self.params['train']:
            self.player_areas_old = len(board.get_player_areas(self.player_name))
            if len(self.memory) > 0:
                self.replay_memory(self.memory, self.params['batch_size'])

        # get results of previous round and train network (if its not the first round)
        if self.num_of_turns != 0 and self.params['train']:
            reward = self.set_reward(self.state_old, state_new)
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
                state_old_tensor = torch.tensor(state_new.reshape((1, 44)), dtype=torch.float32).to(DEVICE)
                prediction = self(state_old_tensor)
                final_move_index = np.argmax(prediction.detach().cpu().numpy()[0])
                final_move = np.eye(6)[final_move_index]
                self.final_move_old = final_move
                self.num_of_model_predictions += 1
                self.final_moves_cache.append(final_move_index)

                # if model predicted the worst (artificially filled) value, select the worst attack
                if len(attacks) <= final_move_index:
                    min_prob = 0
                    min_prob_index = 0
                    for i, attack in enumerate(attacks):
                        prob = probability_of_successful_attack(board, attack[0].get_name(), attack[1].get_name())
                        if min_prob > prob:
                            min_prob = prob
                            min_prob_index = i
                    
                    final_move = np.eye(6)[min_prob_index]
                    self.final_move_old = final_move
                    final_move_index = min_prob_index
                    self.bad_prediction = True
                    self.num_of_bad_predictions += 1



        src_target = attacks[final_move_index]
        dprint("[{}] Attack: {} ({}) -> {} ({}), prob. of success: {} (index {}, attacks len: {})".format("NN" if NN_predicted else "RD", src_target[0].get_name(), src_target[0].get_dice(), src_target[1].get_name(), src_target[1].get_dice(), probability_of_successful_attack(board, src_target[0].get_name(), src_target[1].get_name()), final_move_index, len(attacks)))
        self.player_areas_old = len(board.get_player_areas(self.player_name))
        self.num_of_turns += 1
        self.performed_attacks += 1
        return BattleCommand(src_target[0].get_name(), src_target[1].get_name())  
    
    def save_ai_state(self):
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
            try:
                pickle.dump(state, config_dictionary_file)
            except Exception as e:
                dprint(" --- Saving config error ---")
                dprint(e)

        with open(os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/pickles/DQN_MEMORY.pickle'), 'wb') as memory_file:
            try:
                pickle.dump(self.memory, memory_file)
            except Exception as e:
                dprint(" --- Saving memory error --- ")
                dprint(e)

    def load_ai_state(self):
        config = None
        memory = None
        if os.path.exists(os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/pickles/DQN_STATE.pickle')):
            with open(os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/pickles/DQN_STATE.pickle'), 'rb') as config_dictionary_file:
                try:
                    config = pickle.load(config_dictionary_file)
                except Exception as e:
                    dprint("--- Config load corrupt ---")
                    dprint(e)

        if os.path.exists(os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/pickles/DQN_MEMORY.pickle')):
            with open(os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/pickles/DQN_MEMORY.pickle'), 'rb') as memory_file:
                try:
                    memory = pickle.load(memory_file)
                except Exception as e:
                    dprint("--- Memory load corrupt ---")
                    dprint(e)

        return config, memory

    def network(self):
        # Layers
        self.f1 = nn.Linear(44, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 6)

        # weights
        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            dprint("Weights were loaded")

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x
    
    def get_state(self, board: Board, moves_this_turn):
        state = []
        area_dices = []

        enemy_areas = []
        for player_name in self.players_order:
            if player_name != self.player_name:
                enemy_areas.extend(board.get_player_areas(player_name))

        my_areas = board.get_player_areas(self.player_name)

        my_areas_ratio = len(my_areas) / (len(enemy_areas) + len(my_areas))

        for area in my_areas:
            area_dices.append(area.get_dice())
        
        area_dices = np.array(area_dices)
        areas_dices_mean = area_dices.mean()

        state.append(my_areas_ratio)
        state.append(areas_dices_mean)

        attacks = [a for a in possible_attacks(board, self.player_name) if a[0].get_dice() > (a[1].get_dice() - (0 if moves_this_turn else 1))]
        # attacks_sorted = sorted(attacks, key=lambda x: probability_of_successful_attack(board, x[0].get_name(), x[1].get_name()), reverse=True)

        # shuffle attacks so network can adopt to variety of inputs
        attacks_sorted = attacks
        np.random.shuffle(attacks_sorted)
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
                
        for i in range(len(state), 44, 7):
            state.append(0.00)
            state.append(1.00)
            state.append(1.00)
            state.append(8.00)
            state.append(45.00)
            state.append(2.00)
            state.append(8.00)

        return np.asarray(state), attacks_sorted

    def set_reward(self, state_old, state_new):
        
        area_ratio_old = state_old[0]
        area_ration_current = state_new[0]

        self.reward = 0
        if self.bad_prediction:
            self.reward = -5
        elif area_ration_current < area_ratio_old:
            self.reward = -5
        elif area_ration_current > area_ratio_old:
            self.reward = 10

        return self.reward

    #  Store the <state, action, reward, next_state, is_done> tuple in a memory buffer for replay memory. 
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # Replay memory and train on it
    def replay_memory(self, memory, batch_size):
        
        # if mem len is lesser than batch_size, there is no point in learning
        if batch_size > len(memory):
            return

        dprint("Starting replaying memory, mem len: {}".format(len(memory)))
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory

        for state, action, reward, next_state in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            
            q_eval = self.forward(state_tensor)
            q_next = self.forward(next_state_tensor)
            q_target = reward + self.gamma * torch.max(q_next[0])

            target_f = q_eval.clone()
            target_f[0][np.argmax(action)] = q_target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(q_eval, target_f)
            loss.backward()
            self.epoch_loss.append(loss.item())
            self.optimizer.step()    

        self.loss_vals.append(sum(self.epoch_loss)/len(self.epoch_loss))
        dprint("Replay finished")       

    # find the most valuable transfer
    def get_transfer_to_border_custom(self, board: Board, player_name, threshold):
        border_names = [a.get_name() for a in board.get_player_border(player_name)]
        borders_arrays = [[] for i in range(len(border_names))] # [ [{1}, {2}, {3, 4, 5} ...], [{1}, ...]]

        for i, border_name in enumerate(border_names):
            closed_set = set(border_names)
            area_ids = []  
            for j in range(0, 6): 
                if j == 0:  
                    area_ids = [border_name]
                    borders_arrays[i].append({border_name})

                areas_temp = set()
                for area in area_ids:
                    closed_set.add(area)
                    area_obj: Area = board.get_area(area)
                    adjacent_areas = [ name for name in area_obj.get_adjacent_areas_names() if board.get_area(name).get_owner_name() == player_name and name not in closed_set]
                    areas_temp.update(adjacent_areas)
                
                areas_temp = areas_temp.difference(area_ids)
                if len(areas_temp) == 0: break
                
                area_ids = areas_temp.copy()
                
                borders_arrays[i].append(areas_temp.copy())
        
        borders_arrays.sort(key=lambda x: max([board.get_area(i).get_dice() for i in board.get_area(list(x[0])[0]).get_adjacent_areas_names() if board.get_area(i).get_owner_name() != player_name]) - board.get_area(list(x[0])[0]).get_dice(), reverse=True)
        
        # iterate over every border
        for most_endangered_border in borders_arrays:
            # iterate over each layer between border and connected areas
            prev_neighs = set()
            for level, neighs in enumerate(most_endangered_border):
                # 0 layer = actual border area
                if level == 0:
                    prev_neighs = neighs.copy()
                    continue
                

                dices = []
                ids = []
                # iterate over neighbours
                for neigh in neighs:
                    ids.append(neigh)
                    area = board.get_area(neigh)
                    dices.append(area.get_dice())
                
                # find dice with highest value
                dices = np.array(dices)
                max_dice_index = np.argmax(dices)

                if dices[max_dice_index] >= threshold:
                    # create intersection with selected area and previous layer of areas
                    new_set = prev_neighs.intersection(board.get_area(ids[max_dice_index]).get_adjacent_areas_names())
                    target_id = random.choice(list(new_set))
                    dprint(f"Sending {dices[max_dice_index]} from {ids[max_dice_index]} to {target_id}")
                    return (ids[max_dice_index], target_id)

                prev_neighs = neighs.copy()
        
        return None
