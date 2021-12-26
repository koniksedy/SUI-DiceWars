
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dicewars.ai.kb.move_selection import get_transfer_from_endangered, get_transfer_to_border
from dicewars.ai.utils import possible_attacks, probability_of_holding_area, probability_of_successful_attack
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.client.game.area import Area
from dicewars.client.game.board import Board

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG = True

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
        self.weights_path = os.path.join(os.getcwd(), 'dicewars/ai/kb/xreinm00/weights/weights-final.h5')
        self.gamma = 0.9      
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
    
    def __del__(self):
        dprint("Total turns: {}, Model made {} predictions, {} of them were bad.".format(self.num_of_turns, self.num_of_model_predictions, self.num_of_bad_predictions))

    def ai_turn(self, board: Board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        
        # move dices to borders
        if nb_transfers_this_turn + 2 < self.max_transfers:
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])

        state, attacks = self.get_state(board)
        
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
            return EndTurnCommand()

        self.bad_prediction = False
        with torch.no_grad():
            state_tensor = torch.tensor(state.reshape((1, 36)), dtype=torch.float32).to(DEVICE)
            prediction = self(state_tensor)
            final_move = np.argmax(prediction.detach().cpu().numpy()[0])
            self.num_of_model_predictions += 1

            # if classifier precited move out of max index, it needs to be corrected manually
            # this shouldn't happen at any time
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

        dprint("[NN] Attack: {} ({}) -> {} ({}), prob. of success: {} {} (index {}, attacks len: {})".format(src_target[0].get_name(), src_target[0].get_dice(), src_target[1].get_name(), src_target[1].get_dice(), probability_of_successful_attack(board, src_target[0].get_name(), src_target[1].get_name()), "[Bad prediction]" if self.bad_prediction else "", final_move, len(attacks)))
        return BattleCommand(src_target[0].get_name(), src_target[1].get_name())  
    
    def get_state(self, board: Board):
        state = []
        attacks = [a for a in possible_attacks(board, self.player_name) if a[0].get_dice() >= a[1].get_dice()]
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
                for area_name in dst.get_adjacent_areas_names():
                    target_area: Area = board.get_area(area_name)
                    if target_area.get_owner_name() != self.player_name:
                        if max_dice_around_target < target_area.get_dice():
                            max_dice_around_target = target_area.get_dice()

                num_of_enemies_around = 0
                for area in areas_around:
                    if board.get_area(area).get_owner_name() != self.player_name: 
                        num_of_enemies_around += 1
                
                state.append(prob_of_success)
                state.append(prob_of_hodl)
                state.append(num_of_enemies_around)
                state.append(max_dice_around_target)
                state.append(src.get_dice())
                state.append(dst.get_dice())
                

                # print("""
                # Training dato:
                # ({})   Src dice:       {},
                # ({})   Target dice:    {},
                #        Enemies around: {},
                #        Success prob.:  {},
                #        Hodl prob.   :  {}
                #  """.format(src.get_name(), src.get_dice(), dst.get_name(), dst.get_dice(), num_of_enemies_around, prob_of_success, prob_of_hodl))

        # fill the rest of the state vector with extreme values -> they shouldn't be selected by model ever
        for i in range(len(state), 36, 6):
            state.append(0.00)
            state.append(1.00)
            state.append(1)
            state.append(8)
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
        self.f1 = nn.Linear(36, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 6)
        # load weights
        self.model = self.load_state_dict(torch.load(self.weights_path))
        dprint("Champion's weights loaded")