import json
import random
import string
import time
from typing import Literal

from bfs import find_path_to_target_adjacent_bidirectional
from pydantic import BaseModel, model_validator
from termcolor import colored


class UnitId(BaseModel):
    id: str

class Position(BaseModel):
    x: int
    y: int

class MoveAction(BaseModel):
    action_type: Literal['move'] = 'move'
    player_id: bool
    unit_id: UnitId
    position: Position

class AttackAction(BaseModel):
    action_type: Literal['attack'] = 'attack'
    player_id: bool
    unit_id: UnitId
    target_id: UnitId

class EndTurnAction(BaseModel):
    action_type: Literal['end_turn'] = 'end_turn'
    player_id: bool

class Unit(BaseModel):
    id: UnitId
    position: Position
    health: int
    max_health: int
    attack: int
    hit_rate: float
    movement_range: int
    max_movement_range: int
    has_attacked: bool
    player_id: bool

    @model_validator(mode='after')
    def validate_unit(self):
        assert self.movement_range <= self.max_movement_range
        assert self.health > 0
        assert self.health <= self.max_health
        assert self.attack > 0
        assert self.hit_rate > 0
        assert self.hit_rate <= 1
        assert self.movement_range > 0
        assert self.max_movement_range > 0
        assert self.id.id.startswith('1' if self.player_id else '0')
        return self
    
    def visual_repr(self):
        return colored(f"{self.id.id[1:]}", 'red' if self.player_id else 'blue')

WorldAction = MoveAction | AttackAction | EndTurnAction


class WorldState(BaseModel):
    turn: bool
    width: int
    height: int
    units: list[Unit]

    def get_unit(self, unit_id: UnitId) -> Unit:
        return next(u for u in self.units if u.id == unit_id)
    
    @model_validator(mode='after')
    def validate_units(self):
        assert len(self.units) == len(set(u.id.id for u in self.units))
        return self

class World:
    def __init__(self, state: WorldState):
        self.state = state
        self.last_action: WorldAction | None = None

    def is_valid_action(self, action: WorldAction) -> bool:
        if action.player_id != self.state.turn:
                return False

        if isinstance(action, MoveAction):
            # Unit must exist
            try:
                unit = self.state.get_unit(action.unit_id)
            except StopIteration:
                return False
            if unit.player_id != action.player_id:
                return False
            # Movement can't be zero 
            if action.position.x == unit.position.x and action.position.y == unit.position.y:
                return False
            # Target cell has to be empty
            if any(u.position.x == action.position.x and u.position.y == action.position.y for u in self.state.units):
                return False
            # Position must be within the world bounds
            if not 0 <= action.position.x < self.state.width and 0 <= action.position.y < self.state.height:
                return False
            # Position must be within the unit's movement range
            if not unit.movement_range >= abs(unit.position.x - action.position.x) + abs(unit.position.y - action.position.y):
                return False
            return True

        elif isinstance(action, AttackAction):
            # Unit must exist
            try:
                unit = self.state.get_unit(action.unit_id)
            except StopIteration:
                return False
            if unit.player_id != action.player_id:
                return False
            if unit.has_attacked:
                return False
            # Target must exist
            try:
                target = self.state.get_unit(action.target_id)
            except StopIteration:
                return False
            if target.player_id == action.player_id:
                return False
            # Target must be within the unit's attack range
            if not 1 == abs(unit.position.x - target.position.x) + abs(unit.position.y - target.position.y):
                return False
            return True
        elif isinstance(action, EndTurnAction):
            return True
        return ValueError(f"Invalid action type: {action}")

    def action(self, action: WorldAction) -> WorldState:
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")
        self.last_action = action

        if isinstance(action, MoveAction):
            unit = self.state.get_unit(action.unit_id)
            distance = abs(unit.position.x - action.position.x) + abs(unit.position.y - action.position.y)
            unit.position = action.position
            unit.movement_range -= distance
            return self.state
        elif isinstance(action, AttackAction):
            unit = self.state.get_unit(action.unit_id)
            target = self.state.get_unit(action.target_id)
            if random.random() < unit.hit_rate:
                target.health -= unit.attack
                if target.health <= 0:
                    self.state.units.remove(target)
            unit.has_attacked = True
            return self.state
        elif isinstance(action, EndTurnAction):
            self.state.turn = not self.state.turn
            # Refresh units for the new player
            for unit in self.state.units:
                unit.movement_range = unit.max_movement_range
                unit.has_attacked = False
            return self.state
        raise ValueError(f"Invalid action: {action}")

    def is_game_over(self) -> bool:
        units_player_1 = [u for u in self.state.units if u.player_id]
        units_player_2 = [u for u in self.state.units if not u.player_id]
        return len(units_player_1) == 0 or len(units_player_2) == 0
    
    def get_winner(self) -> bool:
        assert self.is_game_over()
        units_player_1 = [u for u in self.state.units if u.player_id]
        units_player_2 = [u for u in self.state.units if not u.player_id]
        if len(units_player_1) == 0:
            return False
        if len(units_player_2) == 0:
            return True
        raise ValueError("Game is not over")
    
    def visualize(self):
        print(f"Last action:", repr(self.last_action))
        # Create a grid representation
        width = self.state.width
        height = self.state.height
        
        # Initialize empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Fill in units
        for unit in self.state.units:
            x, y = unit.position.x, unit.position.y
            
            # Calculate health bar (using 8ths)
            health_ratio = unit.health / 100  # Assuming max health is 100
            blocks = round(health_ratio * 8)
            health_bar = '█' * blocks + '░' * (8 - blocks)
            
            # Use different symbols for different players
            symbol = unit.visual_repr()
            grid[y][x] = symbol
            
        # Print the grid
        print('  ' + ' '.join(str(i) for i in range(width)))  # Column headers
        for y in range(height):
            print(f"{y} {'|'.join(grid[y])}")  # Row with index
            
        # Print unit details
        print(colored(f"Turn: {'Player 1' if self.state.turn else 'Player 0'}", 'red' if self.state.turn else 'blue'))
        print("\nUnits:")
        for unit in self.state.units:
            health_ratio = unit.health / unit.max_health
            blocks = round(health_ratio * 8)
            health_bar = '█' * blocks + '░' * (8 - blocks)
            player = "Player 1" if unit.player_id else "Player 0"
            print(colored(f"{player} unit {unit.id.id[1:]} at ({unit.position.x},{unit.position.y}): {health_bar} ({unit.health}hp) | Attack: {unit.attack} | Movement: {unit.movement_range}/{unit.max_movement_range} | Has attacked: {unit.has_attacked}", 'red' if unit.player_id else 'blue'))


class GenerateWorld0():
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def generate(self) -> World:
        units = []
        n_units_0 = random.randint(1, 4)
        n_units_1 = random.randint(1, 4)
        
        ids = random.sample(list(set(string.ascii_letters.upper())), n_units_0 + n_units_1)
        for i in range(n_units_0):
            unit_id = UnitId(id=f"0{ids[i]}".upper())
            units.append(Unit(id=unit_id, position=Position(x=random.randint(0, self.width - 1), y=random.randint(0, self.height - 1)), health=5, attack=1, movement_range=1, max_movement_range=1, has_attacked=False, player_id=False, max_health=5, hit_rate=0.5))
        for i in range(n_units_1):
            unit_id = UnitId(id=f"1{ids[n_units_0 + i]}".upper())
            units.append(Unit(id=unit_id, position=Position(x=random.randint(0, self.width - 1), y=random.randint(0, self.height - 1)), health=5, attack=1, movement_range=1, max_movement_range=1, has_attacked=False, player_id=True, max_health=5, hit_rate=0.5))
        print([u.id.id for u in units])

        return World(state=WorldState(width=self.width, height=self.height, units=units, turn=True))


class Engine():
    def get_action(self, world: World) -> WorldAction:
        raise NotImplementedError()


class GreedyEngine(Engine):
    def get_action(self, world: World) -> WorldAction:
        world_state = world.state
        units = world_state.units
        units_mine = [u for u in units if u.player_id == world_state.turn]
        units_opponent = [u for u in units if u.player_id != world_state.turn]

        
        for unit_mine in units_mine:
            # First try to attack the closest opponent
            if unit_mine.has_attacked:
                continue
            for unit_opponent in units_opponent:
                attack_action = AttackAction(player_id=world_state.turn, unit_id=unit_mine.id, target_id=unit_opponent.id)
                if world.is_valid_action(attack_action):
                    return attack_action
        
        
            # Else try to move to the closest opponent
            if unit_mine.movement_range == 0:
                continue
            closest_dist = float('inf')
            closest_pos = None
            for unit_opponent in units_opponent:
                pos, dist = self.find_closest_position(unit_mine, unit_opponent, world_state)
                if pos is None:
                    continue
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = pos
            if closest_pos:
                move_action = MoveAction(player_id=world_state.turn, unit_id=unit_mine.id, position=closest_pos)
                if world.is_valid_action(move_action):
                    return move_action

        return EndTurnAction(player_id=world_state.turn)
        

    def find_closest_position(self, unit_mine: Unit, unit_opponent: Unit, world_state: WorldState) -> tuple[Position | None, int]:
        obstacles = [(u.position.x, u.position.y) for u in world_state.units]
        path = find_path_to_target_adjacent_bidirectional(world_state.width, world_state.height, obstacles, (unit_mine.position.x, unit_mine.position.y), (unit_opponent.position.x, unit_opponent.position.y))
        if path is None:
            return None, 99999
        return Position(x=path[min(unit_mine.movement_range, len(path) - 1)][0], y=path[min(unit_mine.movement_range, len(path) - 1)][1]), max(len(path), unit_mine.movement_range)

class Player():
    def __init__(self, player_id: bool, engine: Engine):
        self.player_id = player_id
        self.engine = engine

    def get_action(self, world: World) -> WorldAction:
        assert world.state.turn == self.player_id
        return self.engine.get_action(world)


class GameLoop():
    def __init__(self, world: World, players: list[Player]):
        self.world = world
        self.players = players

    def run(self):
        i = 0
        while not self.world.is_game_over():
            # time.sleep(.1)
            i += 1
            state_json = self.world.state.model_dump()
            action = self.players[self.world.state.turn].get_action(self.world)
            action_json = action.model_dump()
            self.save_sa_tuple(state_json, action_json)
            self.world.action(action)
            self.world.visualize()
        print("Game over.", "Winner:", colored("Player 1" if self.world.get_winner() else "Player 0", 'red' if self.world.get_winner() else 'blue'))


    def save_sa_tuple(self, state_json: dict, action_json: dict):
        with open("sa_tuples.txt", "a") as f:
            f.write(json.dumps({"state": state_json, "action": action_json}) + "\n")
