import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import r_json
from pologic.game_config import GameConfig
from podecide.game_manager import HumanGameManager
from run.functions import build_single_foldmk

TR_RESULTS_FP = '_models/dmk/training_results.json'
DMK_MODELS_FD = '_models/dmk/'
if __name__ == "__main__":

    logger = get_pylogger(
        name=       f'human_game',
        folder=     DMK_MODELS_FD,
        level=      10,
        #flat_child= True,
    )
    logger.info("Human Game starts")

    loop_results = r_json(TR_RESULTS_FP)
    if loop_results:

        game_config = GameConfig.from_name(folder=DMK_MODELS_FD)
        n_ai_players = game_config.table_size - 1
        logger.info(f"> {game_config.table_size} table players")

        key = 'refs_ranked' if 'refs_ranked' in loop_results else 'dmk_ranked'
        dmks_ranked = loop_results[key]
        dmk_names_ai = dmks_ranked[:n_ai_players]   # bests from ranked
        #dmk_names_ai = [dmks_ranked[1]]             # if you want to use specific player/s
        logger.info(f'got players from loop results: {dmk_names_ai}')
    else:
        game_config = GameConfig.from_yaml('_models/dmk/3players_2bets_gc.yaml')

        n_ai_players = game_config.table_size - 1
        dmk_names_ai = ['dmk073a00']

        for nm in dmk_names_ai:
            build_single_foldmk(
                    game_config=game_config,
                    name='dmk073a00',
                    family='b',
                    model_path='_models/dmk/dmk073a00/dmk073a00.pt',
                    logger=logger
                )


    gm = HumanGameManager(
        dmk_names=      dmk_names_ai,
        game_config=    game_config,
        logger=         logger,
        #debug_dmks=     True,
        debug_tables=   True,
    )
    gm.start_game()