# Jakub Adamčiak
# Bakalárska práca
# Posilňované učenie na hre Bomberman

import pandas as pd

def get_bombs_coords(game_state, unit_id):
    dfh = pd.DataFrame.from_dict(game_state["entities"])
    if ("type" in dfh) and ("unit_id" in dfh):
        dfh = dfh[dfh["type"] == "b"]
        dfhc = dfh[dfh["unit_id"] == unit_id][["x", "y"]].values.tolist()
        if dfhc != []:
            return dfhc
    return None