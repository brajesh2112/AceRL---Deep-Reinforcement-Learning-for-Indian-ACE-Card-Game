#!/usr/bin/env python3
"""
AceRL Agent Training Script - Enhanced for Hidden Information
Train AI agents to play the ACE card game with realistic hidden information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

from config import *
from game_engine import GameEngine, encode_game_state_vector, card_to_index

class EnhancedDQNAgent(nn.Module):
    """Enhanced DQN agent for hidden information card games"""
    
    def __init__(self, state_size, action_size, hidden_size=1024):
        super(EnhancedDQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhanced neural network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, action_size)
        
        # Uncertainty estimation head for hidden information
        self.uncertainty_head = nn.Linear(hidden_size // 4, action_size)
        
        # Regularization
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
        features = F.relu(self.fc3(x))
        
        # Q-values
        q_values = self.fc4(features)
        
        # Uncertainty estimates (for handling hidden information)
        uncertainty = torch.sigmoid(self.uncertainty_head(features))
        
        return q_values, uncertainty
    
    def get_action(self, state, valid_actions, epsilon=0.0):
        """Get action using epsilon-greedy policy with uncertainty consideration"""
        if random.random() < epsilon:
            # Exploration: random valid action
            return random.choice(valid_actions)
        else:
            # Exploitation: best Q-value action considering uncertainty
            with torch.no_grad():
                q_values, uncertainty = self.forward(state)
                
                # Adjust Q-values based on uncertainty (penalize uncertain actions)
                if q_values.dim() > 1:
                    q_values = q_values[0]
                    uncertainty = uncertainty[0]
                
                adjusted_q = q_values - 0.5 * uncertainty  # Balance confidence vs uncertainty
                
                # Mask invalid actions
                masked_q = torch.full((self.action_size,), float('-inf'))
                for action in valid_actions:
                    masked_q[action] = adjusted_q[action]
                
                return masked_q.argmax().item()

class PrioritizedReplayBuffer:
    """Prioritized experience replay for better learning efficiency"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization strength
        self.beta = beta    # Importance sampling
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get maximum priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get experiences
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float),
                torch.stack(next_states), torch.tensor(dones)), indices, torch.tensor(weights, dtype=torch.float)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

def calculate_enhanced_reward(player_id, game_state, card_played, won_trick=False, game_phase="mid"):
    """Enhanced reward calculation considering game phase and strategy"""
    reward = 0.0
    
    # Base reward for winning/losing trick
    if won_trick:
        reward += REWARD_WIN_TRICK
    else:
        reward += REWARD_LOSE_TRICK
    
    # Game phase considerations
    if game_phase == "early":
        # Early game: encourage information gathering
        if card_played.rank in HIGH_RANKS and won_trick:
            reward += PENALTY_HIGH_CARD_EARLY  # Penalty for wasting high cards early
        elif card_played.rank not in HIGH_RANKS:
            reward += 0.3  # Bonus for conservative play
    
    elif game_phase == "late":
        # Late game: encourage aggressive play
        if card_played.rank in HIGH_RANKS and won_trick:
            reward += BONUS_STRATEGIC_LATE
        elif card_played.rank not in HIGH_RANKS and not won_trick:
            reward -= 0.5  # Penalty for being too conservative late
    
    # Strategic bonuses
    player = game_state.players[player_id]
    high_cards_remaining = sum(1 for c in player.hand if c.rank in HIGH_RANKS)
    
    if high_cards_remaining > 3:
        reward += BONUS_HIGH_CARDS_IN_HAND
    
    # Hidden information bonus: reward for making good decisions with limited info
    if hasattr(game_state, 'information_quality'):
        info_bonus = game_state.information_quality * 0.2
        reward += info_bonus
    
    return reward

def encode_hidden_info_state(game_state, moves, player_id):
    """Encode game state with only information visible to the specified player"""
    current_player = game_state.players[player_id]
    
    # Only encode information that the player can see
    # 1. Player's own hand (fully known)
    hand_vector = torch.zeros(NUM_CARDS)
    for card in current_player.hand:
        hand_vector[card_to_index(card)] = 1
    
    # 2. Played cards (public information)
    played_vector = torch.zeros(NUM_CARDS)
    for player in game_state.players:
        for card in player.cards_played:
            played_vector[card_to_index(card)] = 1
    
    # 3. Limited history (recent moves only - what player observed)
    history_vector = torch.zeros(40 * 3)  # Smaller history, less detailed
    recent_moves = moves[-40:] if moves else []
    
    for i, (pid, card, trick_num) in enumerate(recent_moves):
        base_idx = i * 3
        if base_idx + 2 < len(history_vector):
            history_vector[base_idx] = pid / 3.0
            history_vector[base_idx + 1] = card_to_index(card) / NUM_CARDS
            history_vector[base_idx + 2] = trick_num / 13.0
    
    # 4. Observable game context
    context_vector = torch.tensor([
        game_state.current_trick / 13.0,
        len(current_player.hand) / 13.0,
        current_player.tricks_won / 13.0,
        len(game_state.trick_cards) / 4.0,
        1.0 if game_state.lead_suit else 0.0,
        # Opponent visible info (hand sizes, tricks won)
        sum(len(p.hand) for p in game_state.players if p.id != player_id) / 39.0,
        sum(p.tricks_won for p in game_state.players if p.id != player_id) / 13.0
    ])
    
    return torch.cat([hand_vector, played_vector, history_vector, context_vector])

def train_episode_hidden_info(agent, target_agent, optimizer, game_engine, replay_buffer, epsilon):
    """Train agent for one episode with hidden information"""
    game_engine.reset_game()
    episode_reward = 0
    move_count = 0
    
    while not game_engine.game_over:
        current_player_id = game_engine.game_state.current_player
        current_player = game_engine.players[current_player_id]
        
        if not current_player.hand:
            break
        
        if current_player_id == 0:  # AI agent player
            # Get current state with hidden information
            state = encode_hidden_info_state(game_engine.game_state, game_engine.logger.moves, current_player_id)
            
            # Get valid actions
            valid_cards = game_engine.get_valid_moves(current_player_id)
            valid_actions = [card_to_index(card) for card in valid_cards]
            
            # Choose action
            action = agent.get_action(state, valid_actions, epsilon)
            
            # Convert action back to card
            card_to_play = None
            for card in valid_cards:
                if card_to_index(card) == action:
                    card_to_play = card
                    break
            
            if card_to_play:
                # Store current state for experience replay
                current_state = state.clone()
                
                # Determine game phase
                cards_played = sum(len(p.cards_played) for p in game_engine.players)
                if cards_played < 20:
                    game_phase = "early"
                elif cards_played < 40:
                    game_phase = "mid"
                else:
                    game_phase = "late"
                
                # Play the card
                success, _ = game_engine.play_card(current_player_id, card_to_play)
                
                if success:
                    move_count += 1
                    
                    # Check if won the trick
                    won_trick = False
                    if len(game_engine.game_state.trick_cards) == 0:  # Trick completed
                        won_trick = (game_engine.game_state.trick_winner == current_player_id)
                    
                    # Calculate enhanced reward
                    reward = calculate_enhanced_reward(
                        current_player_id, 
                        game_engine.game_state, 
                        card_to_play, 
                        won_trick, 
                        game_phase
                    )
                    episode_reward += reward
                    
                    # Get next state with hidden information
                    if not game_engine.game_over:
                        next_state = encode_hidden_info_state(
                            game_engine.game_state, 
                            game_engine.logger.moves, 
                            current_player_id
                        )
                        done = False
                    else:
                        # Game ended - add final reward bonus/penalty
                        winners = game_engine.get_winner()
                        if winners and current_player_id in [w.id for w in winners]:
                            reward += 5.0  # Win bonus
                        else:
                            reward -= 1.0  # Loss penalty
                        
                        next_state = torch.zeros_like(current_state)
                        done = True
                    
                    # Store experience in prioritized replay buffer
                    replay_buffer.push(
                        current_state,
                        action,
                        reward,
                        next_state,
                        done
                    )
        else:
            # Bot player - use enhanced bot strategy with hidden information
            game_engine.simulate_bot_turn('hard')  # Use hard difficulty for better training
    
    return episode_reward, move_count

def train_dqn_enhanced(agent, target_agent, optimizer, replay_buffer, batch_size=BATCH_SIZE):
    """Enhanced DQN training with prioritized replay and uncertainty loss"""
    batch_data, indices, weights = replay_buffer.sample(batch_size)
    if batch_data is None:
        return None
    
    states, actions, rewards, next_states, dones = batch_data
    
    # Current Q values and uncertainty
    q_values, uncertainty = agent(states)
    current_q_values = q_values.gather(1, actions.long().unsqueeze(1))
    
    # Next Q values from target network
    with torch.no_grad():
        next_q_values, _ = target_agent(next_states)
        next_max_q = next_q_values.max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_max_q * ~dones)
    
    # Compute TD errors for priority update
    td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
    
    # Main Q-learning loss with importance sampling weights
    q_loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
    
    # Uncertainty regularization loss
    uncertainty_loss = uncertainty.mean() * 0.01  # Small regularization
    
    # Total loss
    total_loss = q_loss + uncertainty_loss
    
    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    
    # Update priorities
    replay_buffer.update_priorities(indices, td_errors)
    
    return total_loss.item()

def evaluate_agent_hidden_info(agent, num_games=100):
    """Evaluate agent performance in hidden information mode"""
    agent.eval()
    total_wins = 0
    total_tricks = 0
    
    for _ in range(num_games):
        game = GameEngine()
        
        while not game.game_over:
            current_player_id = game.game_state.current_player
            
            if current_player_id == 0:  # AI agent
                current_player = game.players[current_player_id]
                if current_player.hand:
                    # Use hidden information state encoding
                    state = encode_hidden_info_state(
                        game.game_state, 
                        game.logger.moves, 
                        current_player_id
                    )
                    valid_cards = game.get_valid_moves(current_player_id)
                    valid_actions = [card_to_index(card) for card in valid_cards]
                    
                    action = agent.get_action(state, valid_actions, epsilon=0.0)
                    
                    # Find corresponding card
                    card_to_play = None
                    for card in valid_cards:
                        if card_to_index(card) == action:
                            card_to_play = card
                            break
                    
                    if card_to_play:
                        game.play_card(current_player_id, card_to_play)
            else:
                # Bot player with hidden information constraints
                game.simulate_bot_turn('hard')
        
        # Check results
        winners = game.get_winner()
        agent_tricks = game.players[0].tricks_won
        total_tricks += agent_tricks
        
        if winners and 0 in [w.id for w in winners]:
            total_wins += 1
    
    win_rate = total_wins / num_games
    avg_tricks = total_tricks / num_games
    
    agent.train()
    return win_rate, avg_tricks

def save_enhanced_model(agent, filepath, training_info=None):
    """Save enhanced model with training metadata"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    save_data = {
        'model_state_dict': agent.state_dict(),
        'model_type': 'EnhancedDQNAgent',
        'model_class': agent.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
        'hidden_information_capable': True,
        'uncertainty_estimation': True
    }
    
    if training_info:
        save_data.update(training_info)
    
    torch.save(save_data, filepath)
    print(f"✅ Enhanced model saved to {filepath}")

def load_enhanced_model(agent, filepath):
    """Load enhanced model"""
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
            
            # Handle both old and new format models
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model_info = {
                    'type': checkpoint.get('model_type', 'Unknown'),
                    'hidden_info': checkpoint.get('hidden_information_capable', False),
                    'uncertainty': checkpoint.get('uncertainty_estimation', False)
                }
                print(f"✅ Enhanced model loaded from {filepath}")
                print(f"   Type: {model_info['type']}")
                print(f"   Hidden Info: {model_info['hidden_info']}")
                print(f"   Uncertainty: {model_info['uncertainty']}")
                return True, model_info
            else:
                # Try loading as old format
                agent.load_state_dict(checkpoint, strict=False)
                print(f"✅ Legacy model loaded and adapted from {filepath}")
                return True, {'type': 'Legacy', 'hidden_info': False, 'uncertainty': False}
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False, None
    
    return False, None

def plot_enhanced_training_progress(episode_rewards, win_rates, avg_tricks, losses, uncertainties=None):
    """Enhanced training progress visualization"""
    fig = plt.figure(figsize=(18, 12))
    
    # Create a 3x2 subplot layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Episode rewards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(episode_rewards, alpha=0.7, color='#2E86AB')
    ax1.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), color='#A23B72', linewidth=2)
    ax1.set_title('Episode Rewards (with 50-episode moving average)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Win rates
    ax2 = fig.add_subplot(gs[0, 1])
    episodes = [i * 50 for i in range(len(win_rates))]
    ax2.plot(episodes, win_rates, marker='o', color='#F18F01', linewidth=2, markersize=6)
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Hidden Info Baseline')
    ax2.set_title('Win Rate Over Time (Hidden Information Mode)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Average tricks won
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(episodes, avg_tricks, marker='s', color='#C73E1D', linewidth=2, markersize=6)
    ax3.axhline(y=3.25, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax3.axhline(y=2.6, color='orange', linestyle='--', alpha=0.7, label='Hidden Info Baseline')
    ax3.set_title('Average Tricks Won', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Tricks')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training losses
    ax4 = fig.add_subplot(gs[1, 1])
    if losses:
        ax4.plot(losses, alpha=0.6, color='#3E4E88')
        ax4.plot(np.convolve(losses, np.ones(100)/100, mode='valid'), color='#F39237', linewidth=2)
        ax4.set_title('Training Loss (with 100-step moving average)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
    
    # Uncertainty evolution (if available)
    ax5 = fig.add_subplot(gs[2, 0])
    if uncertainties:
        ax5.plot(uncertainties, color='#9B2226', alpha=0.7)
        ax5.plot(np.convolve(uncertainties, np.ones(50)/50, mode='valid'), color='#005F73', linewidth=2)
        ax5.set_title('Model Uncertainty Over Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Average Uncertainty')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Uncertainty data\nnot available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)
        ax5.set_title('Model Uncertainty', fontsize=12, fontweight='bold')
    
    # Performance summary
    ax6 = fig.add_subplot(gs[2, 1])
    if win_rates:
        # Create performance summary
        final_win_rate = win_rates[-1] if win_rates else 0
        final_tricks = avg_tricks[-1] if avg_tricks else 0
        improvement = (final_win_rate - 0.2) / 0.2 * 100  # vs hidden info baseline
        
        summary_text = f"""TRAINING SUMMARY
        
Final Win Rate: {final_win_rate:.1%}
Final Avg Tricks: {final_tricks:.2f}
vs Hidden Info Baseline: {improvement:+.1f}%
Total Episodes: {len(episode_rewards)}

PERFORMANCE GRADE:
{get_performance_grade(final_win_rate)}"""
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Training Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('AceRL Enhanced Training Progress - Hidden Information Mode', 
                 fontsize=16, fontweight='bold')
    
    # Save plot
    plt.savefig('models/enhanced_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_performance_grade(win_rate):
    """Get performance grade for hidden information mode"""
    if win_rate >= 0.35:
        return "🌟 EXCELLENT (A+)"
    elif win_rate >= 0.3:
        return "⭐ VERY GOOD (A)"
    elif win_rate >= 0.25:
        return "👍 GOOD (B+)"
    elif win_rate >= 0.2:
        return "📚 LEARNING (B)"
    else:
        return "🎯 DEVELOPING (C)"

def main():
    """Enhanced main training function with hidden information support"""
    print("🚀 Starting AceRL Enhanced Agent Training (Hidden Information)")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize game environment
    game_engine = GameEngine()
    
    # Calculate state size for hidden information
    # Hand (52) + Played cards (52) + Limited history (40 * 3) + Enhanced context (7)
    state_size = NUM_CARDS + NUM_CARDS + (40 * 3) + 7
    action_size = NUM_CARDS
    
    print(f"🧠 Enhanced Neural Network Configuration:")
    print(f"   State size: {state_size} (optimized for hidden information)")
    print(f"   Action size: {action_size}")
    print(f"   Hidden layers: {HIDDEN_SIZE} → {HIDDEN_SIZE//2} → {HIDDEN_SIZE//4}")
    print(f"   Special features: Uncertainty estimation, Prioritized replay")
    
    # Initialize enhanced agent and target network
    agent = EnhancedDQNAgent(state_size, action_size, HIDDEN_SIZE)
    target_agent = EnhancedDQNAgent(state_size, action_size, HIDDEN_SIZE)
    
    # Try to load existing model
    model_loaded, model_info = load_enhanced_model(agent, BEST_MODEL_PATH)
    if model_loaded:
        print("🔄 Continuing training from existing model")
        if model_info:
            print(f"   Model type: {model_info.get('type', 'Unknown')}")
            print(f"   Hidden info capable: {model_info.get('hidden_info', False)}")
    else:
        print("🆕 Starting fresh training with enhanced architecture")
    
    # Copy weights to target network
    target_agent.load_state_dict(agent.state_dict())
    
    # Initialize optimizer with enhanced settings
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Initialize prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer(MEMORY_SIZE)
    
    # Enhanced training parameters
    num_episodes = NUM_EPISODES
    epsilon = EPSILON
    target_update_freq = TARGET_UPDATE_FREQUENCY
    eval_freq = EVALUATION_FREQUENCY
    save_freq = SAVE_FREQUENCY
    
    # Training statistics
    episode_rewards = []
    win_rates = []
    avg_tricks_list = []
    losses = []
    uncertainties = []
    best_win_rate = 0.0
    
    print(f"\n🎯 Enhanced Training Configuration:")
    print(f"   Episodes: {num_episodes}")
    print(f"   Evaluation: Every {eval_freq} episodes")
    print(f"   Saving: Every {save_freq} episodes")
    print(f"   Target update: Every {target_update_freq} episodes")
    print(f"   Replay buffer: Prioritized with {MEMORY_SIZE:,} capacity")
    print(f"   Game mode: Hidden information (realistic)")
    
    try:
        print(f"\n🏃‍♂️ Starting training loop...")
        for episode in range(1, num_episodes + 1):
            # Train one episode with hidden information
            episode_reward, moves = train_episode_hidden_info(
                agent, target_agent, optimizer, game_engine, replay_buffer, epsilon
            )
            episode_rewards.append(episode_reward)
            
            # Enhanced training with prioritized replay
            if len(replay_buffer) > BATCH_SIZE:
                loss = train_dqn_enhanced(agent, target_agent, optimizer, replay_buffer)
                if loss is not None:
                    losses.append(loss)
                    
                    # Track model uncertainty
                    if episode % 50 == 0:
                        with torch.no_grad():
                            sample_state = torch.randn(1, state_size)
                            _, uncertainty = agent(sample_state)
                            avg_uncertainty = uncertainty.mean().item()
                            uncertainties.append(avg_uncertainty)
            
            # Update target network
            if episode % target_update_freq == 0:
                target_agent.load_state_dict(agent.state_dict())
            
            # Enhanced epsilon decay
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            
            # Progress logging with enhanced metrics
            if episode % 100 == 0:
                recent_reward = np.mean(episode_rewards[-100:])
                recent_loss = np.mean(losses[-100:]) if losses else 0
                recent_uncertainty = np.mean(uncertainties[-10:]) if uncertainties else 0
                
                print(f"Episode {episode:4d}: "
                      f"Reward={recent_reward:6.2f}, "
                      f"Loss={recent_loss:.6f}, "
                      f"Uncertainty={recent_uncertainty:.3f}, "
                      f"ε={epsilon:.4f}, "
                      f"Buffer={len(replay_buffer):,}")
            
            # Enhanced evaluation
            if episode % eval_freq == 0:
                print(f"\n📈 Evaluation at episode {episode}...")
                win_rate, avg_tricks = evaluate_agent_hidden_info(agent, num_games=EVAL_GAMES)
                
                win_rates.append(win_rate)
                avg_tricks_list.append(avg_tricks)
                
                # Performance assessment
                hidden_info_baseline = 0.2  # Adjusted baseline for hidden information
                improvement = (win_rate - hidden_info_baseline) / hidden_info_baseline * 100
                
                print(f"   Win Rate: {win_rate:.1%} ({improvement:+.1f}% vs hidden info baseline)")
                print(f"   Average Tricks: {avg_tricks:.2f}")
                print(f"   Performance Grade: {get_performance_grade(win_rate)}")
                
                if losses:
                    print(f"   Recent Loss: {np.mean(losses[-100:]):.6f}")
                if uncertainties:
                    print(f"   Model Uncertainty: {np.mean(uncertainties[-5:]):.3f}")
                
                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    training_info = {
                        'episode': episode,
                        'win_rate': win_rate,
                        'avg_tricks': avg_tricks,
                        'improvement_vs_baseline': improvement,
                        'total_episodes_trained': episode,
                        'final_epsilon': epsilon
                    }
                    save_enhanced_model(agent, BEST_MODEL_PATH, training_info)
                    print(f"   🏆 New best model! Win rate: {win_rate:.1%}")
            
            # Regular saving
            if episode % save_freq == 0:
                checkpoint_path = f'models/enhanced_checkpoint_episode_{episode}.pth'
                save_enhanced_model(agent, checkpoint_path)
                print(f"   💾 Enhanced checkpoint saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation (Hidden Information Mode)...")
    final_win_rate, final_avg_tricks = evaluate_agent_hidden_info(agent, num_games=FINAL_EVAL_GAMES)
    
    # Calculate final improvements
    hidden_baseline_improvement = (final_win_rate - 0.2) / 0.2 * 100
    standard_baseline_improvement = (final_win_rate - 0.25) / 0.25 * 100
    
    print(f"\n🏁 ENHANCED TRAINING COMPLETED")
    print(f"{'=' * 70}")
    print(f"🎮 MODE: Hidden Information (Realistic)")
    print(f"🤖 AGENT: {agent.__class__.__name__}")
    print(f"")
    print(f"📊 FINAL RESULTS:")
    print(f"   Final Win Rate: {final_win_rate:.1%}")
    print(f"   Final Average Tricks: {final_avg_tricks:.2f}")
    print(f"   Best Win Rate Achieved: {best_win_rate:.1%}")
    print(f"   Episodes Trained: {len(episode_rewards):,}")
    print(f"")
    print(f"📈 PERFORMANCE vs BASELINES:")
    print(f"   vs Hidden Info Baseline (20%): {hidden_baseline_improvement:+.1f}%")
    print(f"   vs Standard Baseline (25%): {standard_baseline_improvement:+.1f}%")
    print(f"   Final Grade: {get_performance_grade(final_win_rate)}")
    
    # Save final model
    final_training_info = {
        'final_episode': len(episode_rewards),
        'final_win_rate': final_win_rate,
        'final_avg_tricks': final_avg_tricks,
        'best_win_rate': best_win_rate,
        'total_training_time': datetime.now().isoformat(),
        'improvement_vs_hidden_baseline': hidden_baseline_improvement,
        'improvement_vs_standard_baseline': standard_baseline_improvement
    }
    save_enhanced_model(agent, 'models/final_enhanced_ace_rl_agent.pth', final_training_info)
    
    # Generate enhanced training plots
    if win_rates:
        print(f"\n📊 Generating enhanced training visualization...")
        plot_enhanced_training_progress(episode_rewards, win_rates, avg_tricks_list, losses, uncertainties)
    
    # Final performance assessment
    print(f"\n🎖️ FINAL ASSESSMENT:")
    if final_win_rate >= 0.35:
        print("🌟 OUTSTANDING: Your agent is a hidden information master!")
        print("   • Significantly outperforms all baselines")
        print("   • Excellent strategic decision making")
        print("   • Ready for advanced challenges")
    elif final_win_rate >= 0.3:
        print("⭐ EXCELLENT: Strong performance in hidden information mode!")
        print("   • Consistently beats baseline opponents")
        print("   • Good strategic understanding")
        print("   • Consider tournament play")
    elif final_win_rate >= 0.25:
        print("👍 GOOD: Solid improvement over baselines!")
        print("   • Shows clear learning progress")
        print("   • Handles hidden information well")
        print("   • Try longer training for even better results")
    elif final_win_rate >= 0.2:
        print("📚 LEARNING: Making progress in a challenging mode!")
        print("   • Hidden information adds significant difficulty")
        print("   • Consider training for more episodes")
        print("   • Fine-tune hyperparameters if needed")
    else:
        print("🎯 DEVELOPING: Keep training to improve performance!")
        print("   • Hidden information mode is challenging")
        print("   • Try adjusting learning rate or network size")
        print("   • Consider curriculum learning approach")
    
    print(f"\n✅ Enhanced training complete!")
    print(f"🎮 Launch dashboard with: python ace_dashboard.py")
    print(f"📁 Models saved in: models/")
    print(f"📊 Training plots: models/enhanced_training_progress.png")

if __name__ == "__main__":
    main()