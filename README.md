# WAI (World AI)

A reinforcement learning experiment for optimizing strategies in a grid-based, turn-based tactical game.

## Game Mechanics

The game is implemented in `world.py` with the following key features:

- Grid-based world with configurable dimensions
- Two players (Player 0 and Player 1) competing against each other
- Units with basic turn-based combat properties

### Action space
Players can perform three types of actions:
1. Move units within their movement range
2. Attack adjacent enemy units
3. End their turn

### World Generation
- Random world generation with configurable width/height
- Random unit placement for both players
- Customizable number of units per player (1-4 units)

## Training Infrastructure

### Data Collection (`test_world.py`)
- Generates gameplay data using a greedy baseline agent
- Records state-action pairs for supervised learning

### Model Training (`test_train.py`)
- Uses Qwen2-0.5B as the base model
- Fine-tunes on collected gameplay data
- Features:
  - BF16 precision training
  - Cosine learning rate scheduling
  - Live sampling during training for quality monitoring
  - Train/test split for evaluation

### Inference (`test_inference.py`)
- Loads fine-tuned models
- Processes game states into model-compatible format
- Generates next actions based on current game state

## Next Steps

### 1. Behavioral Cloning
- [ ] Collect larger dataset from expert gameplay
- [ ] Implement better data augmentation
- [ ] Add action masking for invalid moves
- [ ] Evaluate action prediction accuracy

### 2. Rejection Sampling
- [ ] Implement temperature-based sampling
- [ ] Add validity checks for sampled actions
- [ ] Create feedback loop for action quality

### 3. Self-Play
- [ ] Create self-play training loop
- [ ] Implement ELO rating system
- [ ] Save and version successful agents
- [ ] Add exploration strategies

### 4. Evaluations
- [ ] Create benchmark scenarios
- [ ] Implement metrics:
  - Win rate vs baseline
  - Average game length
  - Action efficiency
  - Strategic positioning score
- [ ] Add visualization tools for strategy analysis
