from world import GameLoop, GenerateWorld0, GreedyEngine, Player, World

world = GenerateWorld0(width=10, height=10).generate()
# world.visualize()

player_1 = Player(player_id=False, engine=GreedyEngine())
player_2 = Player(player_id=True, engine=GreedyEngine())

game_loop = GameLoop(world, [player_1, player_2])
game_loop.run()