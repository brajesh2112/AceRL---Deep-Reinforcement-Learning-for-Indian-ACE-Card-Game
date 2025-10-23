"""
Fixed AceRL Game Engine - Core game logic with proper round transitions
Handles the fundamental ACE card game mechanics correctly
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Card:
    """Represents a playing card"""
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    
    def __repr__(self):
        return f"{self.rank} of {self.suit}"
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        return hash((self.suit, self.rank))
    
    def rank_value(self):
        """Get numerical rank for comparison (Ace=14, King=13, etc.)"""
        rank_values = {
            'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
            '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
        }
        return rank_values.get(self.rank, 0)

class Deck:
    """Standard 52-card deck"""
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
    
    def shuffle(self):
        """Shuffle the deck randomly"""
        random.shuffle(self.cards)
    
    def deal(self, num_players=4):
        """Deal cards evenly to players"""
        hands = [[] for _ in range(num_players)]
        card_idx = 0
        
        # Deal cards round-robin style
        while card_idx < len(self.cards):
            for player_idx in range(num_players):
                if card_idx < len(self.cards):
                    hands[player_idx].append(self.cards[card_idx])
                    card_idx += 1
        
        return hands

class Player:
    """Represents a player in the game"""
    def __init__(self, player_id, hand=None):
        self.id = player_id
        self.hand = hand or []
        self.tricks_won = 0
        self.total_score = 0
        self.cards_played = []
        self.collected_cards = []  # FIXED: Added collected cards tracking
    
    def show_hand(self):
        """Return string representation of hand"""
        if not self.hand:
            return "No cards"
        return ', '.join(str(card) for card in self.hand)
    
    def play_card(self, card):
        """Play a card from hand"""
        if card in self.hand:
            self.hand.remove(card)
            self.cards_played.append(card)
            return card
        return None
    
    def has_suit(self, suit):
        """Check if player has cards of given suit"""
        return any(card.suit == suit for card in self.hand)
    
    def get_valid_plays(self, lead_suit=None):
        """Get list of valid cards to play - FIXED for Indian ACE rules"""
        if not lead_suit:
            # Leading - can play any card
            return self.hand.copy()
        
        # Must follow suit if possible
        suit_cards = [card for card in self.hand if card.suit == lead_suit]
        if suit_cards:
            return suit_cards
        else:
            # Can't follow suit - can play any card
            return self.hand.copy()
    
    def find_ace_of_spades(self):
        """Check if player has Ace of Spades"""
        return any(card.suit == "Spades" and card.rank == "A" for card in self.hand)

class GameState:
    """Represents the current state of the game"""
    def __init__(self, players, current_trick=0, current_player=0):
        self.players = players
        self.current_trick = current_trick
        self.current_player = current_player
        self.trick_cards = []       # Cards played in current trick
        self.trick_winner = None
        self.lead_suit = None
        self.game_over = False
    
    def is_game_over(self):
        """Check if game is finished - FIXED for Indian ACE rules"""
        # Game ends when any player empties their hand
        return any(len(player.hand) == 0 for player in self.players)
    
    def get_current_player(self):
        """Get the current player"""
        return self.players[self.current_player]
    
    def advance_player(self):
        """Move to next player"""
        self.current_player = (self.current_player + 1) % len(self.players)
    
    def start_new_trick(self):
        """Initialize a new trick"""
        self.current_trick += 1
        self.trick_cards = []
        self.trick_winner = None
        self.lead_suit = None

class GameLogger:
    """Logs game events for analysis and replay"""
    def __init__(self):
        self.moves = []  # (player_id, card, trick_number, timestamp)
        self.trick_winners = []
        self.game_events = []
    
    def log_move(self, player_id, card, trick_number):
        """Log a card play"""
        self.moves.append((player_id, card, trick_number))
    
    def log_trick_winner(self, player_id, trick_number):
        """Log trick winner"""
        self.trick_winners.append((player_id, trick_number))
    
    def log_event(self, event_type, data):
        """Log general game event"""
        self.game_events.append((event_type, data))
    
    def get_move_history(self, max_moves=None):
        """Get recent move history"""
        if max_moves:
            return self.moves[-max_moves:]
        return self.moves.copy()

def card_to_index(card):
    """Convert card to numerical index for ML models"""
    suit_idx = SUITS.index(card.suit)
    rank_idx = RANKS.index(card.rank)
    return suit_idx * len(RANKS) + rank_idx

def index_to_card(index):
    """Convert numerical index back to card"""
    suit_idx = index // len(RANKS)
    rank_idx = index % len(RANKS)
    return Card(SUITS[suit_idx], RANKS[rank_idx])

def encode_hand_vector(hand):
    """Encode hand as binary vector for ML models"""
    vector = torch.zeros(NUM_CARDS)
    for card in hand:
        vector[card_to_index(card)] = 1
    return vector

def encode_game_state_vector(game_state, move_history, max_history=40):
    """Encode complete game state for ML models"""
    current_player = game_state.get_current_player()
    
    # Encode current hand
    hand_vector = encode_hand_vector(current_player.hand)
    
    # Encode move history
    history_vector = torch.zeros(max_history * 5)
    recent_moves = move_history[-max_history:] if move_history else []
    
    for i, (player_id, card, trick_num) in enumerate(recent_moves):
        base_idx = i * 5
        if base_idx + 4 < len(history_vector):
            history_vector[base_idx] = player_id / 3.0  # Normalize player ID
            history_vector[base_idx + 1] = card_to_index(card) / NUM_CARDS  # Normalize card
            history_vector[base_idx + 2] = trick_num / 13.0  # Normalize trick
            history_vector[base_idx + 3] = RANKS.index(card.rank) / len(RANKS)  # Rank value
            history_vector[base_idx + 4] = SUITS.index(card.suit) / len(SUITS)  # Suit value
    
    # Encode current game context
    context_vector = torch.tensor([
        game_state.current_trick / 13.0,  # Game progress
        len(current_player.hand) / 13.0,  # Cards remaining ratio
        len(current_player.collected_cards) / 52.0,  # FIXED: Use collected cards
        len(game_state.trick_cards) / 4.0,  # Trick progress
        1.0 if game_state.lead_suit else 0.0  # Has lead suit
    ])
    
    return torch.cat([hand_vector, history_vector, context_vector])

def rank_value(rank):
    """Get numerical value of card rank for comparison"""
    return RANKS.index(rank)

def determine_trick_winner(trick_cards, lead_suit):
    """FIXED: Determine winner of a trick according to Indian ACE rules"""
    if not trick_cards:
        return None, []
    
    # Find highest card of lead suit
    lead_suit_cards = [(pid, card) for pid, card in trick_cards if card.suit == lead_suit]
    
    if lead_suit_cards:
        # Winner is highest card of lead suit
        winner_id, winning_card = max(lead_suit_cards, key=lambda x: x[1].rank_value())
    else:
        # No one followed suit (shouldn't happen in normal play)
        # Fallback to first player
        winner_id, winning_card = trick_cards[0]
    
    # CRITICAL FIX: Return ALL cards in the trick to winner
    all_cards = [card for _, card in trick_cards]
    return winner_id, all_cards

def calculate_game_score(players):
    """Calculate final game scores for Indian ACE"""
    scores = {}
    for player in players:
        scores[player.id] = {
            'cards_left': len(player.hand),
            'cards_collected': len(player.collected_cards),
            'total_cards_played': len(player.cards_played),
            'high_cards_played': len([c for c in player.cards_played if c.rank in HIGH_RANKS]),
            'is_winner': len(player.hand) == 0
        }
    return scores

class BotStrategy:
    """AI strategy for bot players - FIXED for Indian ACE"""
    
    @staticmethod
    def select_card_basic(player, game_state, difficulty='medium'):
        """Basic bot strategy with difficulty levels"""
        valid_cards = player.get_valid_plays(game_state.lead_suit)
        
        if not valid_cards:
            return None
        
        if difficulty == 'easy':
            # Random play
            return random.choice(valid_cards)
        
        elif difficulty == 'medium':
            # Simple heuristics for Indian ACE
            if game_state.lead_suit:
                # Following suit
                same_suit_cards = [c for c in valid_cards if c.suit == game_state.lead_suit]
                if same_suit_cards:
                    if game_state.trick_cards:
                        # Check if we can win cheaply
                        highest_so_far = max([card for _, card in game_state.trick_cards 
                                            if card.suit == game_state.lead_suit], 
                                           key=lambda x: x.rank_value(), default=None)
                        if highest_so_far:
                            # Try to play just under or lowest
                            under_cards = [c for c in same_suit_cards 
                                         if c.rank_value() < highest_so_far.rank_value()]
                            if under_cards:
                                return max(under_cards, key=lambda x: x.rank_value())  # Highest under
                    
                    # Play lowest card of suit to avoid collecting
                    return min(same_suit_cards, key=lambda x: x.rank_value())
                else:
                    # Can't follow suit - CRITICAL: Play HIGHEST cards to get rid of them
                    return max(valid_cards, key=lambda x: x.rank_value())
            else:
                # Leading - play medium value card
                medium_cards = [c for c in valid_cards if c.rank not in HIGH_RANKS and c.rank not in ['2', '3']]
                if medium_cards:
                    return random.choice(medium_cards)
                return min(valid_cards, key=lambda x: x.rank_value())
        
        else:  # hard
            # Advanced strategy for Indian ACE
            return BotStrategy._advanced_indian_ace_strategy(player, game_state, valid_cards)
    
    @staticmethod
    def _advanced_indian_ace_strategy(player, game_state, valid_cards):
        """Advanced bot strategy for Indian ACE"""
        cards_left_in_hand = len(player.hand)
        collected_cards = len(player.collected_cards)
        
        if game_state.lead_suit:
            # Following suit
            same_suit_cards = [c for c in valid_cards if c.suit == game_state.lead_suit]
            if same_suit_cards:
                if game_state.trick_cards:
                    highest_played = max([card for _, card in game_state.trick_cards 
                                        if card.suit == game_state.lead_suit],
                                       key=lambda x: x.rank_value(), default=None)
                    
                    if highest_played:
                        winning_cards = [c for c in same_suit_cards 
                                       if c.rank_value() > highest_played.rank_value()]
                        
                        if winning_cards:
                            # Decide whether to win based on hand size and cards collected
                            if cards_left_in_hand <= 3:  # Close to winning
                                # Only win if we must or with lowest winning card
                                return min(winning_cards, key=lambda x: x.rank_value())
                            elif collected_cards > 10:  # Already collected too many
                                # Avoid winning unless forced
                                return min(same_suit_cards, key=lambda x: x.rank_value())
                            else:
                                # Strategic win with lowest possible
                                return min(winning_cards, key=lambda x: x.rank_value())
                
                # Play lowest card of suit to avoid collecting
                return min(same_suit_cards, key=lambda x: x.rank_value())
            else:
                # Can't follow suit - CRITICAL: Play highest cards to empty hand
                if cards_left_in_hand <= 5:  # Endgame
                    return max(valid_cards, key=lambda x: x.rank_value())
                else:
                    # Play high cards, but not the absolute highest unless necessary
                    high_cards = [c for c in valid_cards if c.rank_value() >= 10]
                    if high_cards:
                        return max(high_cards, key=lambda x: x.rank_value())
                    else:
                        return max(valid_cards, key=lambda x: x.rank_value())
        
        else:
            # Leading the trick
            if cards_left_in_hand <= 3:  # Close to winning
                # Play conservatively to avoid collecting more
                low_cards = [c for c in valid_cards if c.rank_value() <= 8]
                if low_cards:
                    return max(low_cards, key=lambda x: x.rank_value())
            
            # Standard leading strategy - medium cards
            medium_cards = [c for c in valid_cards if 6 <= c.rank_value() <= 10]
            if medium_cards:
                return random.choice(medium_cards)
            
            # Only high or low cards left
            return min(valid_cards, key=lambda x: x.rank_value())

class GameEngine:
    """Main game engine that orchestrates gameplay - FIXED for proper round flow"""
    
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.reset_game()
    
    def reset_game(self):
        """Initialize a new game"""
        # Create and shuffle deck
        deck = Deck()
        deck.shuffle()
        hands = deck.deal(self.num_players)
        
        # Create players
        self.players = [Player(i, hands[i]) for i in range(self.num_players)]
        self.game_state = GameState(self.players)
        self.logger = GameLogger()
        
        # Game tracking
        self.tricks_played = 0
        self.game_over = False
        
        # FIXED: Find who has Ace of Spades to start
        for i, player in enumerate(self.players):
            if player.find_ace_of_spades():
                self.game_state.current_player = i
                break
    
    def play_card(self, player_id, card):
        """FIXED: Process a card play with proper flow control"""
        if self.game_over:
            return False, "Game is over"
        
        player = self.players[player_id]
        if player_id != self.game_state.current_player:
            return False, f"Not player {player_id}'s turn"
        
        # Validate card play
        valid_cards = player.get_valid_plays(self.game_state.lead_suit)
        if card not in valid_cards:
            return False, "Invalid card play"
        
        # Play the card
        played_card = player.play_card(card)
        if not played_card:
            return False, "Card not in player's hand"
        
        # Add to current trick
        self.game_state.trick_cards.append((player_id, played_card))
        
        # Set lead suit if first card
        if not self.game_state.lead_suit:
            self.game_state.lead_suit = played_card.suit
        
        # Log the move
        self.logger.log_move(player_id, played_card, self.game_state.current_trick + 1)
        
        # Check if trick is complete
        if len(self.game_state.trick_cards) == self.num_players:
            self._complete_trick()
        else:
            # Move to next player
            self.game_state.advance_player()
        
        # Check if game is over
        if self.game_state.is_game_over():
            self.game_over = True
            self._finalize_game()
        
        return True, "Card played successfully"
    
    def _complete_trick(self):
        """FIXED: Complete the current trick properly"""
        # Determine winner using the FIXED function
        winner_id, all_cards = determine_trick_winner(self.game_state.trick_cards, self.game_state.lead_suit)
        
        if winner_id is not None:
            # CRITICAL FIX: Award ALL cards from trick to winner
            self.players[winner_id].collected_cards.extend(all_cards)
            self.players[winner_id].tricks_won += 1  # Also increment tricks won
            self.game_state.trick_winner = winner_id
            
            # Log trick winner
            self.logger.log_trick_winner(winner_id, self.game_state.current_trick + 1)
            
            # CRITICAL FIX: Winner leads next trick
            self.game_state.current_player = winner_id
        
        # Start next trick
        self.game_state.start_new_trick()
        self.tricks_played += 1
    
    def _finalize_game(self):
        """Finalize the game and calculate final scores"""
        self.final_scores = calculate_game_score(self.players)
        self.logger.log_event("game_end", self.final_scores)
    
    def get_winner(self):
        """FIXED: Get the game winner(s) for Indian ACE"""
        if not self.game_over:
            return None
        
        # Winner is first to empty hand
        winners = [player for player in self.players if len(player.hand) == 0]
        return winners
    
    def get_game_summary(self):
        """Get comprehensive game summary"""
        return {
            'players': {
                p.id: {
                    'cards_left': len(p.hand), 
                    'cards_collected': len(p.collected_cards),
                    'tricks_won': p.tricks_won
                } for p in self.players
            },
            'current_trick': self.game_state.current_trick,
            'tricks_played': self.tricks_played,
            'current_player': self.game_state.current_player,
            'game_over': self.game_over,
            'winner': [p.id for p in self.get_winner()] if self.get_winner() else None
        }
    
    def simulate_bot_turn(self, difficulty='medium'):
        """Simulate a bot player's turn"""
        if self.game_over:
            return False, "Game is over"
        
        current_player = self.game_state.get_current_player()
        if not current_player.hand:
            # No cards left - check if game should end
            if self.game_state.is_game_over():
                self.game_over = True
                self._finalize_game()
            return False, "Player has no cards"
        
        # Use bot strategy to select card
        selected_card = BotStrategy.select_card_basic(current_player, self.game_state, difficulty)
        
        if selected_card:
            return self.play_card(current_player.id, selected_card)
        else:
            return False, "Bot could not select a card"
    
    def get_valid_moves(self, player_id=None):
        """Get valid moves for a player"""
        if player_id is None:
            player_id = self.game_state.current_player
        
        player = self.players[player_id]
        return player.get_valid_plays(self.game_state.lead_suit)

def create_game():
    """Factory function to create a new game"""
    return GameEngine()

if __name__ == "__main__":
    print("🧪 Testing FIXED AceRL Game Engine...")
    
    game = create_game()
    print(f"✅ Game created with {len(game.players)} players")
    
    for player in game.players:
        print(f"Player {player.id}: {len(player.hand)} cards")
        if player.find_ace_of_spades():
            print(f"  -> Has Ace of Spades! (Starts the game)")
    
    print(f"🎯 Starting player: {game.game_state.current_player}")
    print("\n🎮 FIXED Game engine ready with proper round transitions!")