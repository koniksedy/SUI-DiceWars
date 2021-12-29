
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dicewars.ai.utils import possible_attacks, probability_of_holding_area, probability_of_successful_attack
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board
import random


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = False

# debug print
def dprint(msg):
    if DEBUG: print(msg)

class AI(torch.nn.Module):
    """
        Model trained by using Deep Q Learning for predicting best attacks.
        Also combined with transfer heuristics. 
    """
    def __init__(self, player_name, board, players_order, max_transfers):
        super().__init__()
        self.player_name = player_name
        self.players_order = players_order
        self.weights_path = os.path.join(os.getcwd(), 'dicewars/ai/xsedym02/weights/weights-final.h5')
        self.gamma = 0.1     
        self.epsilon = 0
        self.counter_games = 0
        self.performed_attacks = 0
        self.fresh_init = True
        self.max_transfers = max_transfers
        self.reserve_for_evacuation = 2 if 2 < max_transfers else 0
        self.bad_prediction = False
        self.num_of_turns = 0
        self.num_of_model_predictions = 0
        self.num_of_bad_predictions = 0

        self.first_layer = 200        # neurons in the first layer
        self.second_layer = 60        # neurons in the second layer
        self.third_layer = 30         # neurons in the third layer
        self.set_network()
    
    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):

        # move dices to borders
        if nb_transfers_this_turn + 2 < self.max_transfers:
            transfer = self.get_transfer_to_border_custom(board, self.player_name, 4)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])

        state, attacks = self.get_state(board, nb_moves_this_turn)
        
        # Evacuation plan
        if len(attacks) == 0 or self.performed_attacks >= 10:
            if nb_transfers_this_turn < self.max_transfers:
                transfer = get_transfer_from_endangered(board, self.player_name)
                if transfer:
                    return TransferCommand(transfer[0], transfer[1])
                else:
                    transfer = self.get_transfer_to_border_custom(board, self.player_name, 2)
                    if transfer:
                        return TransferCommand(transfer[0], transfer[1])

            self.performed_attacks = 0
            return EndTurnCommand()

        self.bad_prediction = False
        with torch.no_grad():
            state_tensor = torch.tensor(state.reshape((1, 44)), dtype=torch.float32).to(DEVICE)
            prediction = self(state_tensor)
            final_move = np.argmax(prediction.detach().cpu().numpy()[0])
            self.num_of_model_predictions += 1

            if len(attacks) <= np.argmax(prediction.detach().cpu().numpy()[0]):
                max_prob = 0
                max_prob_index = 0
                for i, attack in enumerate(attacks):
                    prob = probability_of_successful_attack(board, attack[0].get_name(), attack[1].get_name())
                    if max_prob < prob:
                        max_prob = prob
                        max_prob_index = i

                final_move = max_prob_index
                self.bad_prediction = True
                self.num_of_bad_predictions += 1
    
        src_target = attacks[final_move]
        self.num_of_turns += 1
        self.performed_attacks += 1

        dprint("[NN] Attack: {} ({}) -> {} ({}), prob. of success: {} (index {}, attacks len: {})".format(src_target[0].get_name(), src_target[0].get_dice(), src_target[1].get_name(), src_target[1].get_dice(), probability_of_successful_attack(board, src_target[0].get_name(), src_target[1].get_name()), final_move, len(attacks)))
        return BattleCommand(src_target[0].get_name(), src_target[1].get_name())  
    
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
                
                state.append(prob_of_success)
                state.append(prob_of_hodl)
                state.append(num_of_enemies_around)
                state.append(max_dice_around_target)
                state.append(sum_dice_around_target)
                state.append(src.get_dice())
                state.append(dst.get_dice())

        # fill the rest of the state vector with extreme values -> they shouldn't be selected by model ever
        for i in range(len(state), 44, 7):
            state.append(0.00)
            state.append(1.00)
            state.append(1)
            state.append(8)
            state.append(45)
            state.append(2)
            state.append(8)

        return np.asarray(state), attacks_sorted

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def set_network(self):
        # Layers
        self.f1 = nn.Linear(44, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 6)
        # load weights
        self.model = self.load_state_dict(torch.load(self.weights_path))
        dprint("Model's weights were loaded...")

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

# From dicewars.ai.kb.move_selection
def areas_expected_loss(board, player_name, areas):
    hold_ps = [probability_of_holding_area(board, a.get_name(), a.get_dice(), player_name) for a in areas]
    return sum((1-p) * a.get_dice() for p, a in zip(hold_ps, areas))


# From dicewars.ai.kb.move_selection
def get_transfer_from_endangered(board, player_name):
    border_names = [a.name for a in board.get_player_border(player_name)]
    all_areas_names = [a.name for a in board.get_player_areas(player_name)]

    retreats = []

    for area in border_names:
        area = board.get_area(area)
        if area.get_dice() < 2:
            continue

        for neigh in area.get_adjacent_areas_names():
            if neigh not in all_areas_names:
                continue
            neigh_area = board.get_area(neigh)

            expected_loss_no_evac = areas_expected_loss(board, player_name, [area, neigh_area])

            src_dice = area.get_dice()
            dst_dice = neigh_area.get_dice()

            dice_moved = min(8-dst_dice, src_dice - 1)

            area.dice -= dice_moved
            neigh_area.dice += dice_moved

            expected_loss_evac = areas_expected_loss(board, player_name, [area, neigh_area])

            area.set_dice(src_dice)
            neigh_area.set_dice(dst_dice)

            retreats.append(((area, neigh_area), expected_loss_no_evac - expected_loss_evac))

    retreats = sorted(retreats, key=lambda x: x[1], reverse=True)

    if retreats:
        retreat = retreats[0]
        if retreat[1] > 0.0:
            return retreat[0][0].get_name(), retreat[0][1].get_name()

    return None
