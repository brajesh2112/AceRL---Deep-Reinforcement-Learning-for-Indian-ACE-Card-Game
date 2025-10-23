"""
Complete Fixed Indian ACE Card Game Dashboard
Special Rule Options: 
1. All same suit = cards disappear, mixed suits = highest wins all
2. Cutting Rule = can't follow suit = cut with highest card and win all
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import json
from datetime import datetime
from functools import partial

# Configuration constants
SUITS = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
RANKS = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
NUM_CARDS = 52

# Game Classes
class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    
    def __repr__(self):
        return f"{self.rank}♥" if self.suit == "Hearts" else f"{self.rank}♦" if self.suit == "Diamonds" else f"{self.rank}♣" if self.suit == "Clubs" else f"{self.rank}♠"
    
    def __eq__(self, other):
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
    def __init__(self):
        self.cards = [Card(s, r) for s in SUITS for r in RANKS]
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self, num_players=4):
        hands = [[] for _ in range(num_players)]
        card_idx = 0
        while card_idx < len(self.cards):
            for i in range(num_players):
                if card_idx < len(self.cards):
                    hands[i].append(self.cards[card_idx])
                    card_idx += 1
        return hands

class Player:
    def __init__(self, player_id, hand):
        self.id = player_id
        self.hand = hand
        self.collected_cards = []  # Cards collected from winning tricks
        self.cards_played = []
    
    def show_hand(self):
        return ', '.join(str(card) for card in self.hand)
    
    def has_suit(self, suit):
        return any(card.suit == suit for card in self.hand)
    
    def get_valid_plays(self, lead_suit=None):
        """Get valid cards based on Indian ACE rules"""
        if not lead_suit:
            # Leading - can play any card
            return self.hand.copy()
        
        # Must follow suit if possible
        suit_cards = [card for card in self.hand if card.suit == lead_suit]
        if suit_cards:
            return suit_cards
        else:
            # Can't follow suit - can play any card (cutting)
            return self.hand.copy()
    
    def find_ace_of_spades(self):
        """Check if player has Ace of Spades"""
        return any(card.suit == "Spades" and card.rank == "A" for card in self.hand)

class GameLogger:
    def __init__(self):
        self.moves = []
        self.trick_winners = []
        self.game_events = []
        self.tricks_history = []
    
    def log_move(self, player_id, card, trick_number):
        self.moves.append((player_id, card, trick_number))
    
    def log_trick_result(self, trick_number, trick_cards, winner_id, all_cards):
        self.tricks_history.append({
            'trick_number': trick_number,
            'trick_cards': trick_cards,
            'winner_id': winner_id,
            'all_cards': all_cards
        })

# Utility Functions
def card_to_idx(card):
    suit_idx = SUITS.index(card.suit)
    rank_idx = RANKS.index(card.rank)
    return suit_idx * len(RANKS) + rank_idx

def encode_hand(hand):
    vec = torch.zeros(NUM_CARDS)
    for card in hand:
        vec[card_to_idx(card)] = 1
    return vec

def encode_game_state(hand, moves, trick_number, collected_cards_count=0, max_history=40):
    """Encode game state for Indian ACE game"""
    # Hand encoding
    hand_vec = encode_hand(hand)
    
    # Move history encoding
    history_vec = torch.zeros(max_history * 5)
    start = max(0, len(moves) - max_history)
    moves_slice = moves[start:]
    
    for i, (player_id, card, trick_num) in enumerate(moves_slice):
        base_idx = i * 5
        if base_idx + 4 < len(history_vec):
            history_vec[base_idx] = player_id / 3.0
            history_vec[base_idx + 1] = card_to_idx(card) / NUM_CARDS
            history_vec[base_idx + 2] = trick_num / 13.0
            history_vec[base_idx + 3] = card.rank_value() / 14.0
            history_vec[base_idx + 4] = SUITS.index(card.suit) / len(SUITS)
    
    # Game context
    context_vec = torch.tensor([
        trick_number / 13.0,
        len(hand) / 13.0 if hand else 0.0,
        collected_cards_count / 52.0,
        0.0,
        0.0
    ])
    
    return torch.cat([hand_vec, history_vec, context_vec])

def determine_trick_winner(trick_cards, lead_suit):
    """
    SPECIAL INDIAN ACE RULE:
    - If ALL 4 cards are same suit: Cards disappear (no winner gets them)  
    - If suits are mixed: Player with highest card gets ALL cards
    """
    if not trick_cards or len(trick_cards) != 4:
        return None, []
    
    # Extract all cards and their suits
    all_cards = [card for _, card in trick_cards]
    all_suits = [card.suit for card in all_cards]
    unique_suits = set(all_suits)
    
    print(f"DEBUG: Trick cards: {[str(card) for card in all_cards]}")
    print(f"DEBUG: Suits: {all_suits}")
    print(f"DEBUG: Unique suits count: {len(unique_suits)}")
    
    # Check if ALL cards are the same suit
    if len(unique_suits) == 1:
        # SPECIAL RULE: All same suit - cards disappear!
        print(f"✨ All cards same suit ({list(unique_suits)[0]}) - cards disappear!")
        return None, []  # No winner, no cards collected
    
    else:
        # MIXED SUITS: Find player with highest card value (any suit)
        winner_id, winning_card = max(trick_cards, key=lambda x: x[1].rank_value())
        print(f"🏆 Mixed suits - {winning_card} (Player {winner_id}) wins all cards!")
        return winner_id, all_cards

def determine_trick_winner_with_cutting(trick_cards, lead_suit):
    """
    CORRECT CUTTING RULE:
    - If someone can't follow suit, they "cut" with any card
    - CUTTING PLAYER gets ALL cards played so far
    - Remaining players SKIP their turn (don't play cards)
    - Cutting player gets more cards (disadvantage for winning)
    """
    if not trick_cards:
        return None, [], False, []
    
    print(f"DEBUG CUTTING: Analyzing trick with {len(trick_cards)} cards, lead suit: {lead_suit}")
    
    # Check if anyone cut (played different suit when they should follow)
    cutting_players = []
    following_players = []
    
    for player_id, card in trick_cards:
        if card.suit != lead_suit:
            cutting_players.append((player_id, card))
            print(f"DEBUG CUTTING: Player {player_id} CUT with {card}")
        else:
            following_players.append((player_id, card))
    
    if cutting_players:
        # Someone cut - find highest cutting card
        cutter_id, cutting_card = max(cutting_players, key=lambda x: x[1].rank_value())
        all_cards_played = [card for _, card in trick_cards]
        
        # Calculate which players were skipped
        cards_played = len(trick_cards)
        players_skipped = []
        
        # Find next players who would have played but now skip
        last_player_id = trick_cards[-1][0]  # Player who cut
        for i in range(4 - cards_played):
            skip_player = (last_player_id + 1 + i) % 4
            players_skipped.append(skip_player)
        
        print(f"🔥 Player {cutter_id} CUTS with {cutting_card}!")
        print(f"   Gets {len(all_cards_played)} cards: {[str(c) for c in all_cards_played]}")
        print(f"   Players {players_skipped} skip their turn")
        
        return cutter_id, all_cards_played, True, players_skipped
    
    else:
        # No cutting yet - continue normal play or complete trick
        if len(trick_cards) == 4:
            # Normal completed trick
            winner_id, winning_card = max(following_players, key=lambda x: x[1].rank_value())
            all_cards = [card for _, card in trick_cards]
            print(f"✅ Normal trick: {winning_card} (Player {winner_id}) wins")
            return winner_id, all_cards, False, []
        else:
            # Trick incomplete, continue
            return None, [], False, []

# AI Agent Class
class IndianAceAgent(nn.Module):
    """AI Agent for Indian ACE game"""
    
    def __init__(self, state_size, action_size, hidden_size=1024):
        super(IndianAceAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, action_size)
        
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_indian_ace_suggestions(self, state, hand, lead_suit=None, trick_cards=[], is_leading=False, top_k=3, cutting_rule=False):
        """Get strategic suggestions for Indian ACE game"""
        try:
            self.eval()
            with torch.no_grad():
                # Get valid cards based on Indian ACE rules
                valid_cards = self._get_valid_cards_indian_ace(hand, lead_suit)
                
                if not valid_cards:
                    return []
                
                # Apply Indian ACE strategy
                suggestions = []
                
                for card in valid_cards:
                    strategic_score = self._calculate_indian_ace_score(
                        card, hand, lead_suit, trick_cards, is_leading, cutting_rule
                    )
                    
                    recommendation = self._get_indian_ace_recommendation(
                        card, lead_suit, trick_cards, is_leading, cutting_rule
                    )
                    
                    suggestions.append({
                        'card': card,
                        'strategic_score': strategic_score,
                        'rank_value': card.rank_value(),
                        'recommendation': recommendation,
                        'reasoning': self._get_strategic_reasoning(card, lead_suit, trick_cards, is_leading, cutting_rule)
                    })
                
                # Sort by strategic score
                suggestions.sort(key=lambda x: x['strategic_score'], reverse=True)
                return suggestions[:top_k]
                
        except Exception as e:
            print(f"Error in AI suggestions: {e}")
            # Fallback: return basic suggestions
            valid_cards = [card for card in hand if not lead_suit or card.suit == lead_suit or not any(c.suit == lead_suit for c in hand)]
            if not valid_cards:
                valid_cards = hand[:3] if len(hand) >= 3 else hand
            
            return [{
                'card': card,
                'strategic_score': 0.5,
                'rank_value': card.rank_value(),
                'recommendation': f"Play {card} (Basic suggestion)",
                'reasoning': "AI fallback - choose strategically"
            } for card in valid_cards]
    
    def _get_valid_cards_indian_ace(self, hand, lead_suit):
        """Get valid cards based on Indian ACE rules"""
        if not lead_suit:
            return hand.copy()
        
        # Must follow suit if possible
        suit_cards = [card for card in hand if card.suit == lead_suit]
        return suit_cards if suit_cards else hand.copy()
    
    def _calculate_indian_ace_score(self, card, hand, lead_suit, trick_cards, is_leading, cutting_rule=False):
        """Calculate strategic score for Indian ACE"""
        score = 0.5  # Base score
        
        if cutting_rule:
            return self._calculate_cutting_score(card, hand, lead_suit, trick_cards, is_leading)
        else:
            return self._calculate_disappear_score(card, hand, lead_suit, trick_cards, is_leading)
    
    def _calculate_cutting_score(self, card, hand, lead_suit, trick_cards, is_leading):
        """Calculate score for cutting rule - cutting gives MORE cards (bad for winning)"""
        score = 0.5
        hand_size = len(hand)
        
        if is_leading:
            # Leading - prefer medium cards
            if 7 <= card.rank_value() <= 10:
                score += 0.4
            elif card.rank_value() >= 12:
                score += 0.2  # High cards OK for leading
            else:
                score += 0.3
        
        elif lead_suit and card.suit == lead_suit:
            # Following suit normally - this is often good
            score += 0.6  # Following is generally safe
            
            if trick_cards:
                # Try not to win unless strategic
                highest_so_far = max([c for _, c in trick_cards if c.suit == lead_suit], 
                                   key=lambda x: x.rank_value(), default=card)
                if card.rank_value() < highest_so_far.rank_value():
                    score += 0.3  # Safe play, won't win
        
        else:
            # CUTTING! This gives you MORE cards (generally bad)
            # Only cut if you have too many cards and need control
            if hand_size >= 8:
                # Lots of cards - cutting might be strategic for control
                score += 0.7
                # Use highest card for cutting
                score += (card.rank_value() / 14.0) * 0.3
            elif hand_size >= 6:
                # Medium cards - cutting is risky
                score += 0.4
                score += (card.rank_value() / 14.0) * 0.2
            else:
                # Few cards - avoid cutting (you'll get more cards!)
                score += 0.2
                # If you must cut, use highest card
                score += (card.rank_value() / 14.0) * 0.1
        
        return min(1.0, max(0.1, score))
    
    def _calculate_disappear_score(self, card, hand, lead_suit, trick_cards, is_leading):
        """Calculate score for disappear rule"""
        score = 0.5
        
        if is_leading:
            # Leading - medium cards preferred
            if 7 <= card.rank_value() <= 10:
                score += 0.3
            elif card.rank_value() >= 13:
                score += 0.1
            else:
                score += 0.2
        
        elif lead_suit and card.suit == lead_suit:
            # Following suit - consider disappear possibility
            suits_so_far = [c.suit for _, c in trick_cards]
            if all(s == lead_suit for s in suits_so_far):
                remaining = 4 - len(trick_cards) - 1
                if remaining == 0:
                    # Cards will disappear - good!
                    score += 0.8
                else:
                    score += 0.5  # Might disappear
            else:
                # Mixed already
                score += 0.4
        
        elif lead_suit:
            # Can't follow suit - might win all cards
            if trick_cards:
                highest = max([c for _, c in trick_cards], key=lambda x: x.rank_value())
                if card.rank_value() > highest.rank_value():
                    # Would win - only good if few cards left
                    score += 0.7 if len(hand) <= 4 else 0.3
                else:
                    score += 0.5
            score += (card.rank_value() / 14.0) * 0.2
        
        return min(1.0, max(0.1, score))
    
    def _get_indian_ace_recommendation(self, card, lead_suit, trick_cards, is_leading, cutting_rule=False):
        """Get recommendation text"""
        if cutting_rule:
            if is_leading:
                return f"LEAD {card} - Set suit"
            elif lead_suit and card.suit == lead_suit:
                return f"FOLLOW {card} - Safe play"
            else:
                return f"CUT with {card} - You'll get MORE cards!"
        else:
            if is_leading:
                return f"LEAD {card} - Control trick"
            elif lead_suit and card.suit == lead_suit:
                suits_played = [c.suit for _, c in trick_cards]
                if all(s == lead_suit for s in suits_played):
                    remaining = 4 - len(trick_cards) - 1
                    if remaining == 0:
                        return f"SAME SUIT {card} - Cards DISAPPEAR!"
                    else:
                        return f"SAME SUIT {card} - Might disappear"
                else:
                    return f"FOLLOW {card} - Mixed suits"
            else:
                return f"DIFFERENT SUIT {card} - Might win all!"
    
    def _get_strategic_reasoning(self, card, lead_suit, trick_cards, is_leading, cutting_rule=False):
        """Get detailed reasoning"""
        if cutting_rule and lead_suit and card.suit != lead_suit:
            return f"CUTTING with {card}: You get ALL cards played (makes winning harder - more cards to get rid of!)"
        
        if is_leading:
            return f"Leading with {card}: Sets the trump suit for this trick."
        elif lead_suit and card.suit == lead_suit:
            return f"Following {lead_suit}: Must play same suit if you have it."
        else:
            return f"Can't follow {lead_suit}: Playing {card} creates mixed suits."

def smart_load_model(model_file):
    """Load model with error handling"""
    if not os.path.exists(model_file):
        return None, None
    
    try:
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            model_type = checkpoint.get('model_type', 'IndianAceAgent')
        else:
            state_dict = checkpoint
            model_type = 'IndianAceAgent'
        
        fc1_weight_shape = state_dict['fc1.weight'].shape
        state_size = fc1_weight_shape[1]
        action_size = NUM_CARDS
        hidden_size = fc1_weight_shape[0]
        
        agent = IndianAceAgent(state_size, action_size, hidden_size)
        agent.load_state_dict(state_dict)
        agent.eval()
        
        print(f"✅ Loaded Indian ACE model: State={state_size}, Hidden={hidden_size}")
        return agent, model_type
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Main Dashboard Class
class IndianAceDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Indian ACE Card Game - AI Strategy Dashboard")
        self.root.geometry("1500x900")
        self.root.configure(bg='#1a1a2e')
        
        # Game state
        self.current_game = None
        self.current_trick_cards = []
        self.lead_suit = None
        self.selected_card = None
        self.game_logger = None
        self.first_game = True
        self.waiting_for_human = False
        self.cutting_rule = False
        
        # AI and stats
        self.agent = None
        self.agent_type = "None"
        self.game_stats = {
            'games_played': 0,
            'games_won': 0,
            'total_cards_collected': 0,
            'best_performance': float('inf'),
            'avg_cards_collected': 0.0
        }
        
        # Colors
        self.colors = {
            'bg_primary': '#1a1a2e',
            'bg_secondary': '#16213e',
            'bg_accent': '#0f3460',
            'text_primary': '#ffffff',
            'text_secondary': '#ecf0f1',
            'accent_blue': '#00d4ff',
            'accent_green': '#00ff88',
            'accent_red': '#ff6b6b',
            'accent_orange': '#ff9800',
            'accent_purple': '#9c27b0',
            'spades': '#2c3e50',
            'hearts': '#e74c3c',
            'diamonds': '#f39c12',
            'clubs': '#27ae60'
        }
        
        self._create_ui()
        self._load_agent()
        self._show_game_rules()
    
    def _create_ui(self):
        """Create the main UI"""
        # Title
        title_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        title_frame.pack(fill='x', pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🃏 Indian ACE Card Game - AI Strategy Dashboard",
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_blue'],
            font=('Arial', 18, 'bold')
        )
        title_label.pack()
        
        # Rule toggle
        rule_frame = tk.Frame(title_frame, bg=self.colors['bg_primary'])
        rule_frame.pack(pady=5)
        
        self.rule_var = tk.BooleanVar()
        rule_check = tk.Checkbutton(
            rule_frame,
            text="🔥 Enable Cutting Rule (can't follow suit = cut and get MORE cards)",
            variable=self.rule_var,
            command=self._toggle_cutting_rule,
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_orange'],
            font=('Arial', 12),
            selectcolor=self.colors['bg_accent']
        )
        rule_check.pack()
        
        self.subtitle_label = tk.Label(
            title_frame,
            text="Same suit = Disappear • Mixed suits = Highest wins all • Empty hand to win!",
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_orange'],
            font=('Arial', 12)
        )
        self.subtitle_label.pack()
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Game area
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'])
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self._create_game_area(left_panel)
        
        # Right panel - AI and stats
        right_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'])
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self._create_ai_panel(right_panel)
    
    def _toggle_cutting_rule(self):
        """Toggle between cutting rule and disappear rule"""
        self.cutting_rule = self.rule_var.get()
        
        if self.cutting_rule:
            self.subtitle_label.config(
                text="🔥 CUTTING: Can't follow suit = Cut and get MORE cards (harder to win)!"
            )
        else:
            self.subtitle_label.config(
                text="Same suit = Disappear • Mixed suits = Highest wins all • Empty hand to win!"
            )
        
        # Reset current game if active
        if self.current_game:
            messagebox.showinfo("Rule Changed", "Rule changed! Start a new game for the change to take effect.")
    
    def _create_game_area(self, parent):
        """Create the main game area"""
        # Controls
        controls_frame = tk.LabelFrame(
            parent,
            text="🎮 Game Controls",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 12, 'bold')
        )
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        btn_frame = tk.Frame(controls_frame, bg=self.colors['bg_secondary'])
        btn_frame.pack(padx=10, pady=10)
        
        buttons = [
            ("🆕 New Game", self._start_new_game, self.colors['accent_green']),
            ("🤖 AI Help", self._get_ai_help, self.colors['accent_blue']),
            ("🎯 Play Card", self._play_card, self.colors['accent_orange']),
            ("📖 Rules", self._show_game_rules, self.colors['accent_purple'])
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(
                btn_frame,
                text=text,
                command=command,
                bg=color,
                fg='white',
                font=('Arial', 11, 'bold'),
                padx=15,
                pady=8
            )
            btn.grid(row=0, column=i, padx=5)
        
        # Current trick display
        trick_frame = tk.LabelFrame(
            parent,
            text="🃏 Current Trick",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 12, 'bold')
        )
        trick_frame.pack(fill='x', padx=10, pady=5)
        
        self.trick_display = tk.Frame(trick_frame, bg=self.colors['bg_secondary'])
        self.trick_display.pack(fill='x', padx=10, pady=10)
        
        # Your hand
        hand_frame = tk.LabelFrame(
            parent,
            text="🃏 Your Hand (Click to select)",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 12, 'bold')
        )
        hand_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.hand_display = tk.Frame(hand_frame, bg=self.colors['bg_secondary'])
        self.hand_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Game log
        log_frame = tk.LabelFrame(
            parent,
            text="📝 Game Log",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 11, 'bold')
        )
        log_frame.pack(fill='x', padx=10, pady=5)
        
        self.game_log = scrolledtext.ScrolledText(
            log_frame,
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary'],
            font=('Consolas', 9),
            height=8
        )
        self.game_log.pack(fill='both', expand=True, padx=10, pady=10)
    
    def _create_ai_panel(self, parent):
        """Create AI suggestions panel"""
        # Agent status
        status_frame = tk.Frame(parent, bg=self.colors['bg_secondary'])
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.agent_status = tk.Label(
            status_frame,
            text="Agent: Loading...",
            bg=self.colors['bg_secondary'],
            fg=self.colors['accent_orange'],
            font=('Arial', 12, 'bold')
        )
        self.agent_status.pack()
        
        # AI suggestions
        ai_frame = tk.LabelFrame(
            parent,
            text="🧠 AI Strategy Suggestions",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 12, 'bold')
        )
        ai_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.suggestions_text = scrolledtext.ScrolledText(
            ai_frame,
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10),
            wrap='word'
        )
        self.suggestions_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Player stats
        stats_frame = tk.LabelFrame(
            parent,
            text="👥 Players Status",
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary'],
            font=('Arial', 11, 'bold')
        )
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.players_display = tk.Frame(stats_frame, bg=self.colors['bg_secondary'])
        self.players_display.pack(fill='x', padx=10, pady=10)
    
    def _load_agent(self):
        """Load AI agent"""
        model_files = [
            'models/best_ace_rl_agent.pth',
            'models/final_ace_rl_agent.pth',
            'models/ace_rl_agent.pth'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                self.agent, self.agent_type = smart_load_model(model_file)
                if self.agent:
                    self.agent_status.config(
                        text=f"✅ Agent: {self.agent_type} (Indian ACE Rules)",
                        fg=self.colors['accent_green']
                    )
                    return
        
        self.agent_status.config(
            text="⚠️ No AI Agent - Basic suggestions only",
            fg=self.colors['accent_red']
        )
    
    def _show_game_rules(self):
        """Show Indian ACE game rules"""
        rules_window = tk.Toplevel(self.root)
        rules_window.title("Indian ACE Game Rules")
        rules_window.geometry("700x700")
        rules_window.configure(bg=self.colors['bg_primary'])
        
        rules_text = scrolledtext.ScrolledText(
            rules_window,
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary'],
            font=('Arial', 12),
            wrap='word'
        )
        rules_text.pack(fill='both', expand=True, padx=20, pady=20)
        
        if self.cutting_rule:
            rules = """🔥 INDIAN ACE CARD GAME - CUTTING RULE

🎯 OBJECTIVE: Be the first player to empty your hand completely!

🔥 CUTTING RULE MECHANICS:
• If you can't follow suit, you "CUT" with any card
• CUTTING player gets ALL cards played so far in the trick
• Remaining players SKIP their turn (don't play cards)
• Cutting player now has MORE cards (harder to win!)
• Cutting is strategic control but comes with penalty

📝 EXAMPLE:
Player 1: 6♣ (leads Clubs)
Player 2: J♣ (follows suit)  
Player 3: A♦ (CAN'T follow Clubs - CUTS with Diamond!)
Player 4: SKIPPED (doesn't play)

Result: Player 3 gets all 3 cards (6♣, J♣, A♦)
Player 3 goes from 5 cards to 8 cards (5+3=8)
Player 4 keeps 6 cards (didn't play)

🎮 GAME FLOW:
1. Player with A♠ starts the game
2. Must follow suit if you have cards of that suit
3. Can't follow suit? You MUST cut and take all cards
4. First to empty hand wins

🧠 STRATEGY:
• Cutting gives control but MORE cards
• Only cut when strategic (many cards, need control)
• Avoid cutting when you have few cards
• Use highest cards for cutting to ensure you win"""
        else:
            rules = """✨ INDIAN ACE CARD GAME - DISAPPEAR RULE

🎯 OBJECTIVE: Be the first player to empty your hand completely!

✨ DISAPPEAR RULE MECHANICS:
• ALL SAME SUIT: Cards disappear (no one gets them)
• MIXED SUITS: Highest card wins ALL cards
• Strategic choice between disappearing vs winning cards

📝 EXAMPLE 1 - Same Suit:
Player 1: 6♣, Player 2: J♣, Player 3: A♣, Player 4: 10♣
Result: All Clubs - Cards DISAPPEAR! (Reduces total cards)

📝 EXAMPLE 2 - Mixed Suits:
Player 1: 6♣, Player 2: J♣, Player 3: A♦, Player 4: 10♣
Result: Mixed suits - A♦ (highest) wins ALL 4 cards

🎮 GAME FLOW:
1. Player with A♠ starts the game
2. Must follow suit if you have cards of that suit
3. Can't follow suit? Play any card (creates mixed suits)
4. First to empty hand wins

🧠 STRATEGY:
• Same suit tricks reduce total cards (good for everyone)
• Mixed suits create big card swings
• Control when to create mixed vs same suit situations"""
        
        rules_text.insert('1.0', rules)
        rules_text.config(state='disabled')
        
        close_btn = tk.Button(
            rules_window,
            text="Let's Play!",
            command=rules_window.destroy,
            bg=self.colors['accent_green'],
            fg='white',
            font=('Arial', 14, 'bold'),
            pady=10
        )
        close_btn.pack(pady=10)
    
    def _start_new_game(self):
        """Start a new Indian ACE game"""
        try:
            # Create and deal cards
            deck = Deck()
            deck.shuffle()
            hands = deck.deal(4)
            
            # Create players
            players = [Player(i, hands[i]) for i in range(4)]
            
            # Find who has Ace of Spades
            starter_id = None
            for i, player in enumerate(players):
                if player.find_ace_of_spades():
                    starter_id = i
                    break
            
            if starter_id is None:
                starter_id = 0  # Fallback
            
            # Initialize game state
            self.current_game = {
                'players': players,
                'current_trick': 0,
                'current_player': starter_id,
                'starter_id': starter_id,
                'game_over': False
            }
            
            # Reset all trick state
            self.current_trick_cards = []
            self.lead_suit = None
            self.selected_card = None
            self.game_logger = GameLogger()
            self.waiting_for_human = False
            
            self._update_displays()
            
            rule_name = "Cutting Rule" if self.cutting_rule else "Disappear Rule"
            
            if starter_id == 0:
                self._log_message("🎯 You have the Ace of Spades! You start the game.")
                self._log_message("💡 Tip: You can play any card to start.")
                self.waiting_for_human = True
            else:
                self._log_message(f"🎮 Player {starter_id + 1} has Ace of Spades and starts.")
                self.waiting_for_human = False
                self.root.after(1000, self._start_bot_turn)
            
            self._log_message(f"🆕 New Indian ACE game started! ({rule_name})")
            
            if self.cutting_rule:
                self._log_message("🔥 CUTTING RULE: Can't follow suit = Cut and get MORE cards!")
            else:
                self._log_message("✨ DISAPPEAR RULE: Same suit = Disappear, Mixed suits = Highest wins all!")
            
            self._log_message("🏆 Goal: Be first to empty your hand!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start game: {e}")
    
    def _update_displays(self):
        """Update all game displays"""
        if not self.current_game:
            return
        
        self._update_hand_display()
        self._update_trick_display()
        self._update_players_display()
    
    def _update_hand_display(self):
        """Update player hand display"""
        # Clear existing display
        for widget in self.hand_display.winfo_children():
            widget.destroy()
        
        if not self.current_game:
            return
        
        player = self.current_game['players'][0]  # Human player
        
        # Sort cards by suit and rank for better display
        sorted_hand = sorted(player.hand, key=lambda c: (c.suit, c.rank_value()))
        
        for i, card in enumerate(sorted_hand):
            # Get card color based on suit
            suit_colors = {
                'Spades': self.colors['spades'],
                'Hearts': self.colors['hearts'],
                'Diamonds': self.colors['diamonds'],
                'Clubs': self.colors['clubs']
            }
            card_color = suit_colors.get(card.suit, self.colors['bg_accent'])
            
            # Create card button
            card_btn = tk.Button(
                self.hand_display,
                text=str(card),
                bg=card_color,
                fg='white',
                font=('Arial', 12, 'bold'),
                width=6,
                height=2,
                command=partial(self._select_card, card)
            )
            
            # Highlight selected card
            if self.selected_card and self.selected_card == card:
                card_btn.config(relief='sunken', bd=4, bg=self.colors['accent_orange'])
            else:
                card_btn.config(relief='raised', bd=2)
            
            # Special highlight for Ace of Spades
            if card.suit == "Spades" and card.rank == "A":
                card_btn.config(bg=self.colors['accent_purple'])
            
            card_btn.grid(row=i//8, column=i%8, padx=2, pady=2)
        
        # Update hand info
        cards_collected = len(player.collected_cards)
        hand_info = f"🃏 Your Hand: {len(player.hand)} cards | Collected: {cards_collected} cards"
        if self.selected_card:
            hand_info += f" | Selected: {self.selected_card}"
        
        hand_frame = self.hand_display.master
        hand_frame.config(text=hand_info)
    
    def _update_trick_display(self):
        """Update current trick display"""
        # Clear existing display
        for widget in self.trick_display.winfo_children():
            widget.destroy()
        
        if not self.current_trick_cards:
            info_label = tk.Label(
                self.trick_display,
                text="No cards played yet in current trick",
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_secondary'],
                font=('Arial', 11)
            )
            info_label.pack()
            return
        
        # Show lead suit info
        if self.lead_suit:
            lead_info = tk.Label(
                self.trick_display,
                text=f"Lead Suit: {self.lead_suit}",
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_blue'],
                font=('Arial', 12, 'bold')
            )
            lead_info.pack()
        
        # Show cards played in this trick
        cards_frame = tk.Frame(self.trick_display, bg=self.colors['bg_secondary'])
        cards_frame.pack(pady=5)
        
        for i, (player_id, card) in enumerate(self.current_trick_cards):
            player_name = "You" if player_id == 0 else f"Bot {player_id}"
            
            # Card display
            suit_colors = {
                'Spades': self.colors['spades'],
                'Hearts': self.colors['hearts'],
                'Diamonds': self.colors['diamonds'],
                'Clubs': self.colors['clubs']
            }
            card_color = suit_colors.get(card.suit, self.colors['bg_accent'])
            
            card_frame = tk.Frame(cards_frame, bg=self.colors['bg_secondary'])
            card_frame.grid(row=0, column=i, padx=5)
            
            player_label = tk.Label(
                card_frame,
                text=player_name,
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                font=('Arial', 10)
            )
            player_label.pack()
            
            card_label = tk.Label(
                card_frame,
                text=str(card),
                bg=card_color,
                fg='white',
                font=('Arial', 12, 'bold'),
                width=6,
                height=2
            )
            card_label.pack(pady=2)
        
        # Show prediction based on current rule
        if len(self.current_trick_cards) > 0:
            self._show_trick_prediction()
    
    def _show_trick_prediction(self):
        """Show prediction based on current rule"""
        if self.cutting_rule:
            # Check if someone cut
            cutting_players = [(pid, card) for pid, card in self.current_trick_cards 
                             if card.suit != self.lead_suit]
            
            if cutting_players:
                highest_cutter_id, highest_cut_card = max(cutting_players, key=lambda x: x[1].rank_value())
                cutter_name = "You" if highest_cutter_id == 0 else f"Bot {highest_cutter_id}"
                
                prediction_info = tk.Label(
                    self.trick_display,
                    text=f"🔥 CUTTING! {cutter_name} wins with {highest_cut_card}",
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['accent_red'],
                    font=('Arial', 11, 'bold')
                )
                prediction_info.pack(pady=5)
                
                remaining = 4 - len(self.current_trick_cards)
                if remaining > 0:
                    skip_info = tk.Label(
                        self.trick_display,
                        text=f"⭐ {remaining} player(s) will be skipped",
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['accent_orange'],
                        font=('Arial', 10)
                    )
                    skip_info.pack()
        
        else:
            # Disappear rule prediction
            all_suits = [card.suit for _, card in self.current_trick_cards]
            unique_suits = set(all_suits)
            
            if len(unique_suits) == 1:
                # All same suit so far
                remaining = 4 - len(self.current_trick_cards)
                if remaining > 0:
                    prediction_info = tk.Label(
                        self.trick_display,
                        text=f"✨ All {list(unique_suits)[0]} so far - cards might DISAPPEAR!",
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['accent_purple'],
                        font=('Arial', 11, 'bold')
                    )
                    prediction_info.pack(pady=5)
            else:
                # Mixed suits already
                highest_card = max([card for _, card in self.current_trick_cards], key=lambda x: x.rank_value())
                highest_player = None
                for pid, card in self.current_trick_cards:
                    if card == highest_card:
                        highest_player = "You" if pid == 0 else f"Bot {pid}"
                        break
                
                prediction_info = tk.Label(
                    self.trick_display,
                    text=f"🏆 Mixed suits - {highest_player} winning with {highest_card}",
                    bg=self.colors['bg_secondary'],
                    fg=self.colors['accent_green'],
                    font=('Arial', 11, 'bold')
                )
                prediction_info.pack(pady=5)
    
    def _update_players_display(self):
        """Update players status display"""
        # Clear existing display
        for widget in self.players_display.winfo_children():
            widget.destroy()
        
        if not self.current_game:
            return
        
        for i, player in enumerate(self.current_game['players']):
            player_name = "You" if i == 0 else f"Bot {i}"
            is_current = (self.current_game['current_player'] == i)
            
            player_frame = tk.Frame(self.players_display, bg=self.colors['bg_accent'])
            player_frame.pack(fill='x', pady=2)
            
            # Player name with current player indicator
            name_text = f"🎯 {player_name}" if is_current else f"   {player_name}"
            name_label = tk.Label(
                player_frame,
                text=name_text,
                bg=self.colors['bg_accent'],
                fg=self.colors['accent_orange'] if is_current else self.colors['text_primary'],
                font=('Arial', 11, 'bold' if is_current else 'normal')
            )
            name_label.pack(side='left', padx=10)
            
            # Stats
            stats_text = f"Hand: {len(player.hand)} | Collected: {len(player.collected_cards)}"
            stats_label = tk.Label(
                player_frame,
                text=stats_text,
                bg=self.colors['bg_accent'],
                fg=self.colors['text_primary'],
                font=('Arial', 10)
            )
            stats_label.pack(side='right', padx=10)
    
    def _select_card(self, card):
        """Select a card for playing"""
        if not self.current_game or self.current_game.get('game_over', False):
            messagebox.showwarning("Warning", "No active game!")
            return
        
        if self.current_game['current_player'] != 0:
            messagebox.showwarning("Warning", "Not your turn!")
            return
        
        player = self.current_game['players'][0]
        if card not in player.hand:
            messagebox.showwarning("Warning", "Card not in your hand!")
            return
        
        # Check if card is valid according to Indian ACE rules
        valid_cards = player.get_valid_plays(self.lead_suit)
        if card not in valid_cards:
            if self.lead_suit:
                messagebox.showwarning(
                    "Invalid Play", 
                    f"Must follow suit ({self.lead_suit}) if you have cards of that suit!"
                )
            return
        
        self.selected_card = card
        self._update_hand_display()
        
        # Show cutting info if applicable
        if self.cutting_rule and self.lead_suit and card.suit != self.lead_suit:
            self._log_message(f"🔥 Selected CUTTING card: {card} - You'll get MORE cards!")
        else:
            self._log_message(f"🎯 Selected: {card} (Value: {card.rank_value()})")
    
    def _get_ai_help(self):
        """Get AI strategy suggestions"""
        if not self.current_game:
            messagebox.showwarning("Warning", "Start a new game first!")
            return
        
        if self.current_game['current_player'] != 0:
            messagebox.showinfo("Info", "Not your turn - wait for bots to play")
            return
        
        try:
            player = self.current_game['players'][0]
            if not player.hand:
                messagebox.showinfo("Info", "No cards remaining!")
                return
            
            # Check if we're leading this trick
            is_leading = len(self.current_trick_cards) == 0
            
            # Get AI suggestions
            if self.agent:
                # Encode current game state
                state = encode_game_state(
                    player.hand,
                    self.game_logger.moves,
                    self.current_game['current_trick'],
                    len(player.collected_cards)
                )
                
                suggestions = self.agent.get_indian_ace_suggestions(
                    state, player.hand, self.lead_suit, self.current_trick_cards, 
                    is_leading, top_k=5, cutting_rule=self.cutting_rule
                )
            else:
                # Fallback suggestions without AI
                suggestions = self._get_fallback_suggestions(player, is_leading)
            
            self._display_ai_suggestions(suggestions, is_leading, player)
            self._log_message("🤖 AI analysis complete - check suggestions panel!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get AI help: {e}")
            print(f"AI Help Error: {e}")
    
    def _get_fallback_suggestions(self, player, is_leading):
        """Get basic suggestions without AI model"""
        valid_cards = player.get_valid_plays(self.lead_suit)
        if not valid_cards:
            return []
        
        suggestions = []
        for card in valid_cards[:5]:
            score = 0.5
            
            if self.cutting_rule and self.lead_suit and card.suit != self.lead_suit:
                # Cutting
                score = 0.3 if len(player.hand) <= 5 else 0.6
                rec = f"CUT with {card} - You'll get MORE cards"
                reason = "Cutting gives you control but more cards to manage"
            elif is_leading:
                score = 0.6 if 7 <= card.rank_value() <= 10 else 0.4
                rec = f"LEAD with {card} - Sets the suit"
                reason = "Leading controls the trick direction"
            else:
                score = 0.5
                rec = f"Play {card} - Follow the rules"
                reason = "Basic play following game rules"
            
            suggestions.append({
                'card': card,
                'strategic_score': score,
                'rank_value': card.rank_value(),
                'recommendation': rec,
                'reasoning': reason
            })
        
        suggestions.sort(key=lambda x: x['strategic_score'], reverse=True)
        return suggestions
    
    def _display_ai_suggestions(self, suggestions, is_leading, player):
        """Display AI suggestions"""
        self.suggestions_text.delete('1.0', tk.END)
        
        if not suggestions:
            self.suggestions_text.insert('1.0', "❌ No valid suggestions available")
            return
        
        rule_name = "CUTTING RULE" if self.cutting_rule else "DISAPPEAR RULE"
        
        content = f"🧠 AI INDIAN ACE STRATEGY ({rule_name})\n"
        content += "=" * 60 + "\n\n"
        
        # Rule explanation
        if self.cutting_rule:
            content += "🔥 CUTTING RULE ACTIVE:\n"
            content += "• Can't follow suit → CUT and get ALL cards 🔥\n"
            content += "• MORE cards = HARDER to win ⚠️\n"
            content += "• Cut strategically for control\n\n"
        else:
            content += "✨ DISAPPEAR RULE ACTIVE:\n"
            content += "• ALL same suit → Cards DISAPPEAR ✨\n"
            content += "• Mixed suits → Highest wins ALL 🏆\n\n"
        
        # Current situation
        if is_leading:
            content += "🎯 SITUATION: You are LEADING this trick\n"
            content += "💡 Your card choice sets the suit others must follow\n\n"
        elif self.lead_suit:
            can_follow = any(card.suit == self.lead_suit for card in player.hand)
            if can_follow:
                content += f"📝 SITUATION: Must follow {self.lead_suit} suit\n"
            else:
                content += f"📝 SITUATION: Can't follow {self.lead_suit} - "
                if self.cutting_rule:
                    content += "You will CUT! 🔥\n"
                else:
                    content += "Creates mixed suits\n"
        
        # Current trick analysis
        if self.current_trick_cards:
            content += f"🃏 Cards played: {len(self.current_trick_cards)}/4\n"
            
            if self.cutting_rule:
                # Check for existing cuts
                cuts = [(pid, card) for pid, card in self.current_trick_cards 
                       if card.suit != self.lead_suit]
                if cuts:
                    cutter_id, cut_card = max(cuts, key=lambda x: x[1].rank_value())
                    cutter_name = "You" if cutter_id == 0 else f"Player {cutter_id+1}"
                    content += f"🔥 {cutter_name} already CUT with {cut_card}\n"
            else:
                # Check suit uniformity
                suits = [card.suit for _, card in self.current_trick_cards]
                if len(set(suits)) == 1:
                    content += f"✨ All {suits[0]} so far - might disappear!\n"
                else:
                    content += f"🏆 Mixed suits - highest card wins all\n"
        
        content += f"\n🎪 Your stats: {len(player.hand)} cards | Collected: {len(player.collected_cards)}\n\n"
        
        # Top suggestions
        content += "🏆 TOP STRATEGIC RECOMMENDATIONS:\n"
        content += "-" * 40 + "\n"
        
        for i, suggestion in enumerate(suggestions[:3], 1):
            card = suggestion['card']
            score = suggestion['strategic_score']
            rec = suggestion['recommendation']
            reason = suggestion.get('reasoning', 'Strategic choice')
            
            content += f"{i}. 🃏 {card} (Value: {card.rank_value()})\n"
            content += f"   Strategy: {rec}\n"
            content += f"   Confidence: {score:.1%}\n"
            content += f"   Reasoning: {reason}\n\n"
        
        # Strategic tips
        content += self._get_strategic_tips_text()
        
        self.suggestions_text.insert('1.0', content)
    
    def _get_strategic_tips_text(self):
        """Get strategic tips based on current rule"""
        if self.cutting_rule:
            return """💡 CUTTING RULE TIPS:
• Cutting gives you MORE cards (harder to win!)
• Only cut when you have many cards and need control
• Use highest cards for cutting to ensure you win
• Avoid cutting when you have few cards
• Balance control vs. card accumulation

"""
        else:
            return """💡 DISAPPEAR RULE TIPS:
• Same suit tricks benefit everyone (fewer cards)
• Mixed suits create big card swings
• High cards are powerful but risky
• Sometimes disappearing cards is better than winning
• Control when to create mixed vs same suit situations

"""
    
    def _play_card(self):
        """Play the selected card"""
        if not self.current_game:
            messagebox.showwarning("Warning", "Start a new game first!")
            return
        
        if self.current_game.get('game_over', False):
            messagebox.showinfo("Info", "Game is over!")
            return
        
        if self.current_game['current_player'] != 0:
            messagebox.showwarning("Warning", "Not your turn!")
            return
        
        if not self.selected_card:
            messagebox.showwarning("Warning", "Select a card first!")
            return
        
        try:
            self._play_human_card()
            self._process_turn_sequence()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play card: {e}")
    
    def _play_human_card(self):
        """Process human player's card"""
        player = self.current_game['players'][0]
        card = self.selected_card
        
        # Validate the play
        valid_cards = player.get_valid_plays(self.lead_suit)
        if card not in valid_cards:
            raise ValueError("Invalid card play")
        
        # Remove card from hand
        player.hand.remove(card)
        
        # Add to current trick
        self.current_trick_cards.append((0, card))
        
        # Set lead suit if first card
        if not self.lead_suit:
            self.lead_suit = card.suit
        
        # Log the play
        trick_num = self.current_game['current_trick'] + 1
        self.game_logger.log_move(0, card, trick_num)
        
        is_cutting = self.cutting_rule and self.lead_suit and card.suit != self.lead_suit
        
        if is_cutting:
            self._log_message(f"🔥 You CUT with: {card}! You'll get all cards and others skip!")
        else:
            self._log_message(f"🎯 You played: {card}")
        
        # Clear selection and advance
        self.selected_card = None
        self.current_game['current_player'] = (self.current_game['current_player'] + 1) % 4
        
        self._update_displays()
    
    def _process_turn_sequence(self):
        """Process turn sequence based on rule"""
        if self.cutting_rule:
            self._process_cutting_sequence()
        else:
            self._process_normal_sequence()
    
    def _process_cutting_sequence(self):
        """Process turns with cutting rule"""
        while not self.current_game.get('game_over', False):
            # Check if someone cut
            if self.current_trick_cards:
                cuts = [(pid, card) for pid, card in self.current_trick_cards 
                       if card.suit != self.lead_suit]
                
                if cuts:
                    # Someone cut - complete trick immediately
                    self._complete_cutting_trick()
                    return
            
            # Check if trick is complete (4 cards)
            if len(self.current_trick_cards) == 4:
                self._complete_cutting_trick()
                return
            
            # Continue with next player
            current_player_id = self.current_game['current_player']
            current_player = self.current_game['players'][current_player_id]
            
            if not current_player.hand:
                if self._check_game_over():
                    return
            
            if current_player_id == 0:
                # Human turn
                self.waiting_for_human = True
                if current_player.hand:
                    self._log_message("🎯 Your turn! Select a card to play.")
                break
            else:
                # Bot turn
                self.waiting_for_human = False
                if current_player.hand:
                    self.root.after(1000, self._start_bot_turn)
                    break
                else:
                    self.current_game['current_player'] = (current_player_id + 1) % 4
    
    def _process_normal_sequence(self):
        """Process turns with normal disappear rule"""
        while not self.current_game.get('game_over', False):
            # Check if trick is complete
            if len(self.current_trick_cards) == 4:
                self._complete_normal_trick()
                return
            
            # Continue with next player
            current_player_id = self.current_game['current_player']
            current_player = self.current_game['players'][current_player_id]
            
            if not current_player.hand:
                if self._check_game_over():
                    return
            
            if current_player_id == 0:
                # Human turn
                self.waiting_for_human = True
                if current_player.hand:
                    self._log_message("🎯 Your turn! Select a card to play.")
                break
            else:
                # Bot turn
                self.waiting_for_human = False
                if current_player.hand:
                    self.root.after(1000, self._start_bot_turn)
                    break
                else:
                    self.current_game['current_player'] = (current_player_id + 1) % 4
    
    def _complete_cutting_trick(self):
        """Complete trick with cutting rule"""
        winner_id, all_cards, was_cut, skipped_players = determine_trick_winner_with_cutting(
            self.current_trick_cards, self.lead_suit
        )
        
        if winner_id is not None:
            # Award cards to winner
            winner = self.current_game['players'][winner_id]
            winner.collected_cards.extend(all_cards)
            
            # Log the result
            cards_str = ", ".join(str(card) for _, card in self.current_trick_cards)
            winner_name = "You" if winner_id == 0 else f"Bot {winner_id}"
            
            if was_cut:
                self._log_message(f"🔥 CUTTING! {winner_name} cuts and gets {len(all_cards)} cards!")
                self._log_message(f"   Cards collected: {cards_str}")
                if skipped_players:
                    skipped_names = []
                    for pid in skipped_players:
                        if pid < len(self.current_game['players']):
                            name = "You" if pid == 0 else f"Bot {pid}"
                            skipped_names.append(name)
                    if skipped_names:
                        self._log_message(f"   Skipped players: {', '.join(skipped_names)}")
            else:
                self._log_message(f"✅ {winner_name} wins {len(all_cards)} cards normally")
                self._log_message(f"   Cards collected: {cards_str}")
            
            # Winner leads next trick
            self.current_game['current_player'] = winner_id
        
        # Move to next trick
        self.current_game['current_trick'] += 1
        self.current_trick_cards = []
        self.lead_suit = None
        
        self._update_displays()
        self._continue_after_trick()
    
    def _complete_normal_trick(self):
        """Complete trick with normal disappear rule"""
        winner_id, all_cards = determine_trick_winner(self.current_trick_cards, self.lead_suit)
        
        cards_str = ", ".join(str(card) for _, card in self.current_trick_cards)
        
        if winner_id is None:
            # Cards disappeared
            suits = [card.suit for _, card in self.current_trick_cards]
            suit_name = suits[0]
            self._log_message(f"✨ ALL {suit_name} - CARDS DISAPPEAR!")
            self._log_message(f"   Disappeared: {cards_str}")
            
            # Find highest card player to lead next
            highest_player_id, _ = max(self.current_trick_cards, key=lambda x: x[1].rank_value())
            self.current_game['current_player'] = highest_player_id
            
        else:
            # Winner gets all cards
            winner = self.current_game['players'][winner_id]
            winner.collected_cards.extend(all_cards)
            
            winner_name = "You" if winner_id == 0 else f"Bot {winner_id}"
            self._log_message(f"🏆 MIXED SUITS: {winner_name} wins {len(all_cards)} cards!")
            self._log_message(f"   Cards collected: {cards_str}")
            
            # Winner leads next trick
            self.current_game['current_player'] = winner_id
        
        # Move to next trick
        self.current_game['current_trick'] += 1
        self.current_trick_cards = []
        self.lead_suit = None
        
        self._update_displays()
        self._continue_after_trick()
    
    def _continue_after_trick(self):
        """Continue game after trick completion"""
        if self._check_game_over():
            return
        
        current_player_id = self.current_game['current_player']
        current_player = self.current_game['players'][current_player_id]
        
        if not current_player.hand:
            if self._check_game_over():
                return
        
        if current_player_id == 0:
            # Human turn
            self.waiting_for_human = True
            if current_player.hand:
                self._log_message("🎯 Your turn to lead! Select a card.")
        else:
            # Bot turn
            self.waiting_for_human = False
            if current_player.hand:
                self.root.after(1500, self._start_bot_turn)
    
    def _start_bot_turn(self):
        """Process bot player's turn"""
        if self.current_game.get('game_over', False):
            return
        
        current_player_id = self.current_game['current_player']
        
        if current_player_id == 0:
            return  # Human turn
        
        player = self.current_game['players'][current_player_id]
        
        if not player.hand:
            if self._check_game_over():
                return
            else:
                self.current_game['current_player'] = (current_player_id + 1) % 4
                self._process_turn_sequence()
                return
        
        # Get valid cards for bot
        valid_cards = player.get_valid_plays(self.lead_suit)
        
        if not valid_cards:
            return
        
        # Bot card selection
        selected_card = self._bot_select_card(player, valid_cards)
        
        if not selected_card:
            selected_card = valid_cards[0]
        
        # Check if cutting
        is_cutting = self.cutting_rule and self.lead_suit and selected_card.suit != self.lead_suit
        
        # Execute play
        player.hand.remove(selected_card)
        self.current_trick_cards.append((current_player_id, selected_card))
        
        # Set lead suit if first card
        if not self.lead_suit:
            self.lead_suit = selected_card.suit
        
        # Log move
        trick_num = self.current_game['current_trick'] + 1
        self.game_logger.log_move(current_player_id, selected_card, trick_num)
        
        if is_cutting:
            self._log_message(f"🔥 Bot {current_player_id} CUTS with: {selected_card}")
        else:
            self._log_message(f"🤖 Bot {current_player_id} played: {selected_card}")
        
        # Advance to next player
        self.current_game['current_player'] = (current_player_id + 1) % 4
        
        self._update_displays()
        self._process_turn_sequence()
    
    def _bot_select_card(self, player, valid_cards):
        """Bot card selection strategy"""
        is_leading = len(self.current_trick_cards) == 0
        
        if self.cutting_rule:
            return self._bot_select_cutting_strategy(player, valid_cards, is_leading)
        else:
            return self._bot_select_disappear_strategy(player, valid_cards, is_leading)
    
    def _bot_select_cutting_strategy(self, player, valid_cards, is_leading):
        """Bot strategy for cutting rule"""
        hand_size = len(player.hand)
        
        if is_leading:
            # Lead with medium cards
            medium_cards = [c for c in valid_cards if 7 <= c.rank_value() <= 10]
            if medium_cards:
                return random.choice(medium_cards)
            return min(valid_cards, key=lambda x: x.rank_value())
        
        elif self.lead_suit and any(c.suit == self.lead_suit for c in player.hand):
            # Can follow suit - generally safer
            suit_cards = [c for c in valid_cards if c.suit == self.lead_suit]
            if suit_cards:
                # Try not to win unless strategic
                if self.current_trick_cards:
                    highest_so_far = max([c for _, c in self.current_trick_cards 
                                        if c.suit == self.lead_suit], 
                                       key=lambda x: x.rank_value(), default=None)
                    if highest_so_far:
                        under_cards = [c for c in suit_cards 
                                     if c.rank_value() < highest_so_far.rank_value()]
                        if under_cards:
                            return max(under_cards, key=lambda x: x.rank_value())
                
                return min(suit_cards, key=lambda x: x.rank_value())
        
        else:
            # Must cut - strategic decision
            if hand_size >= 8:
                # Many cards - cut for control with highest card
                return max(valid_cards, key=lambda x: x.rank_value())
            elif hand_size <= 4:
                # Few cards - avoid cutting, but if must cut, use lowest
                return min(valid_cards, key=lambda x: x.rank_value())
            else:
                # Medium cards - moderate cutting strategy
                return random.choice(valid_cards)
        
        return valid_cards[0]
    
    def _bot_select_disappear_strategy(self, player, valid_cards, is_leading):
        """Bot strategy for disappear rule"""
        if is_leading:
            # Lead with moderate cards
            medium_cards = [c for c in valid_cards if 6 <= c.rank_value() <= 10]
            if medium_cards:
                return random.choice(medium_cards)
            return min(valid_cards, key=lambda x: x.rank_value())
        
        elif self.lead_suit and any(c.suit == self.lead_suit for c in player.hand):
            # Following suit
            suit_cards = [c for c in valid_cards if c.suit == self.lead_suit]
            
            if suit_cards:
                # Check if all same suit so far
                suits_played = [c.suit for _, c in self.current_trick_cards]
                if all(s == self.lead_suit for s in suits_played):
                    # Might disappear - play lower card
                    return min(suit_cards, key=lambda x: x.rank_value())
                else:
                    # Mixed already - be strategic
                    if self.current_trick_cards:
                        highest = max([c for _, c in self.current_trick_cards], 
                                    key=lambda x: x.rank_value())
                        under_cards = [c for c in suit_cards 
                                     if c.rank_value() < highest.rank_value()]
                        if under_cards:
                            return max(under_cards, key=lambda x: x.rank_value())
                    return min(suit_cards, key=lambda x: x.rank_value())
        
        else:
            # Can't follow suit - creates mixed suits
            if self.current_trick_cards:
                highest_so_far = max([c for _, c in self.current_trick_cards], 
                                   key=lambda x: x.rank_value())
                
                # Only win if beneficial (few cards left)
                if len(player.hand) <= 3:
                    winning_cards = [c for c in valid_cards 
                                   if c.rank_value() > highest_so_far.rank_value()]
                    if winning_cards:
                        return min(winning_cards, key=lambda x: x.rank_value())
                
                # Otherwise play high card to get rid of it
                return max(valid_cards, key=lambda x: x.rank_value())
            else:
                return max(valid_cards, key=lambda x: x.rank_value())
        
        return valid_cards[0]
    
    def _check_game_over(self):
        """Check if game is over"""
        for player in self.current_game['players']:
            if len(player.hand) == 0:
                self._end_game()
                return True
        return False
    
    def _end_game(self):
        """End the game and show results"""
        self.current_game['game_over'] = True
        
        # Find winner and results
        winner = None
        results = []
        
        for player in self.current_game['players']:
            results.append({
                'player_id': player.id,
                'cards_left': len(player.hand),
                'cards_collected': len(player.collected_cards),
                'total_cards': len(player.hand) + len(player.collected_cards),
                'is_winner': len(player.hand) == 0
            })
            
            if len(player.hand) == 0:
                winner = player
        
        # Sort results
        results.sort(key=lambda x: (x['cards_left'], x['cards_collected']))
        
        # Update stats
        self.game_stats['games_played'] += 1
        if winner and winner.id == 0:
            self.game_stats['games_won'] += 1
        
        human_collected = len(self.current_game['players'][0].collected_cards)
        self.game_stats['total_cards_collected'] += human_collected
        self.game_stats['avg_cards_collected'] = (
            self.game_stats['total_cards_collected'] / self.game_stats['games_played']
        )
        if human_collected < self.game_stats['best_performance']:
            self.game_stats['best_performance'] = human_collected
        
        self._show_game_results(results, winner)
    
    def _show_game_results(self, results, winner):
        """Show game results"""
        results_window = tk.Toplevel(self.root)
        rule_name = "Cutting Rule" if self.cutting_rule else "Disappear Rule"
        results_window.title(f"Indian ACE Results ({rule_name})")
        results_window.geometry("600x600")
        results_window.configure(bg=self.colors['bg_primary'])
        
        # Title
        title_label = tk.Label(
            results_window,
            text=f"INDIAN ACE RESULTS ({rule_name.upper()})",
            bg=self.colors['bg_primary'],
            fg=self.colors['accent_blue'],
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=10)
        
        # Results display
        results_text = scrolledtext.ScrolledText(
            results_window,
            bg=self.colors['bg_accent'],
            fg=self.colors['text_primary'],
            font=('Consolas', 12),
            height=20
        )
        results_text.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Format results
        if winner:
            content = f"WINNER: Player {winner.id + 1}"
            if winner.id == 0:
                content += " (YOU!) - CONGRATULATIONS!\n"
            else:
                content += f" (Bot)\n"
        else:
            content = "No winner determined\n"
        
        content += "=" * 50 + "\n\n"
        
        content += f"{rule_name.upper()} RESULTS:\n"
        content += "Winner = First to empty hand\n"
        content += "-" * 40 + "\n"
        
        for i, result in enumerate(results, 1):
            pid = result['player_id']
            player_name = "You" if pid == 0 else f"Bot {pid}"
            cards_left = result['cards_left']
            cards_collected = result['cards_collected']
            
            content += f"{i}. {player_name:<8} | "
            content += f"Left: {cards_left:2d} | "
            content += f"Collected: {cards_collected:2d}"
            
            if result['is_winner']:
                content += " - WINNER!\n"
            elif pid == 0:
                content += " (You)\n"
            else:
                content += "\n"
        
        # Performance analysis
        human_result = next(r for r in results if r['player_id'] == 0)
        human_rank = results.index(human_result) + 1
        
        content += "\n" + "=" * 50 + "\n"
        content += "YOUR PERFORMANCE:\n"
        content += f"Rank: {human_rank}/4\n"
        content += f"Cards Left: {human_result['cards_left']}\n"
        content += f"Cards Collected: {human_result['cards_collected']}\n"
        
        if human_rank == 1:
            content += "EXCELLENT! You won!\n"
        elif human_rank == 2:
            content += "Good job! Second place!\n"
        else:
            content += "Room for improvement - try AI suggestions!\n"
        
        # Session stats
        content += f"\nSESSION STATISTICS:\n"
        content += f"Games Played: {self.game_stats['games_played']}\n"
        content += f"Games Won: {self.game_stats['games_won']}\n"
        content += f"Win Rate: {self.game_stats['games_won']/max(1, self.game_stats['games_played']):.1%}\n"
        content += f"Avg Cards Collected: {self.game_stats['avg_cards_collected']:.1f}\n"
        
        results_text.insert('1.0', content)
        results_text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(results_window, bg=self.colors['bg_primary'])
        btn_frame.pack(pady=10)
        
        play_again_btn = tk.Button(
            btn_frame,
            text="Play Again",
            command=lambda: [results_window.destroy(), self._start_new_game()],
            bg=self.colors['accent_green'],
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        play_again_btn.pack(side='left', padx=10)
        
        close_btn = tk.Button(
            btn_frame,
            text="Close",
            command=results_window.destroy,
            bg=self.colors['accent_blue'],
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        close_btn.pack(side='left', padx=10)
    
    def _log_message(self, message):
        """Add message to game log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.game_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.game_log.see(tk.END)
    
    def run(self):
        """Run the dashboard"""
        print("Indian ACE Card Game Dashboard Starting...")
        
        if self.agent:
            print("AI Agent loaded - Ready for strategy suggestions!")
        else:
            print("No AI Agent - Basic suggestions available")
        
        print("Two Game Rule Options:")
        print("1. DISAPPEAR RULE (default): Same suit = disappear, mixed = highest wins")
        print("2. CUTTING RULE (toggle): Can't follow suit = cut and get MORE cards")
        
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Dashboard error: {e}")
            messagebox.showerror("Error", f"Dashboard failed: {e}")

def main():
    """Main function"""
    print("Indian ACE Card Game Dashboard")
    print("=" * 60)
    print("CUTTING RULE IMPLEMENTATION:")
    print("- Player who can't follow suit CUTS with any card")
    print("- Cutting player gets ALL cards played so far")
    print("- Remaining players SKIP their turn")
    print("- More cards = harder to win!")
    print()
    print("DISAPPEAR RULE:")
    print("- All same suit: Cards disappear")
    print("- Mixed suits: Highest card wins all")
    print()
    
    try:
        dashboard = IndianAceDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Failed to launch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()