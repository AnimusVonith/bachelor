def get_arena_templates(name = None):

    arena_templates = {

        "mock_15x15_state" : {
            "game_id": "dev",
            "agents": {
                "a": {"agent_id": "a","unit_ids": ["c"]},   #,"e","g"
                "b": {"agent_id": "b","unit_ids": ["d"]}},  #,"f","h"
            "unit_state": {
                "c": {"coordinates": [4,0],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
                "d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0},
                #"e": {"coordinates": [5,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "e","agent_id": "a","invulnerability": 0},
                #"f": {"coordinates": [9,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "f","agent_id": "b","invulnerability": 0},
                #"g": {"coordinates": [1,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "g","agent_id": "a","invulnerability": 0},
                #"h": {"coordinates": [13,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "h","agent_id": "b","invulnerability": 0}
            },
            "entities": [
                {"created": 0,"x": 14,"y": 1,"type": "m"},{"created": 0,"x": 0,"y": 1,"type": "m"},{"created": 0,"x": 1,"y": 2,"type": "m"},
                {"created": 0,"x": 13,"y": 2,"type": "m"},{"created": 0,"x": 11,"y": 0,"type": "m"},{"created": 0,"x": 3,"y": 0,"type": "m"},
                {"created": 0,"x": 12,"y": 12,"type": "m"},{"created": 0,"x": 2,"y": 12,"type": "m"},{"created": 0,"x": 11,"y": 14,"type": "m"},
                {"created": 0,"x": 3,"y": 14,"type": "m"},{"created": 0,"x": 6,"y": 8,"type": "m"},{"created": 0,"x": 8,"y": 8,"type": "m"},
                {"created": 0,"x": 14,"y": 7,"type": "m"},{"created": 0,"x": 0,"y": 7,"type": "m"},{"created": 0,"x": 2,"y": 4,"type": "m"},
                {"created": 0,"x": 12,"y": 4,"type": "m"},{"created": 0,"x": 8,"y": 14,"type": "m"},{"created": 0,"x": 6,"y": 14,"type": "m"},
                {"created": 0,"x": 6,"y": 7,"type": "m"},{"created": 0,"x": 8,"y": 7,"type": "m"},{"created": 0,"x": 9,"y": 9,"type": "m"},
                {"created": 0,"x": 5,"y": 9,"type": "m"},{"created": 0,"x": 9,"y": 2,"type": "m"},{"created": 0,"x": 5,"y": 2,"type": "m"},
                {"created": 0,"x": 11,"y": 8,"type": "m"},{"created": 0,"x": 3,"y": 8,"type": "m"},{"created": 0,"x": 5,"y": 5,"type": "m"},
                {"created": 0,"x": 9,"y": 5,"type": "m"},{"created": 0,"x": 5,"y": 11,"type": "m"},{"created": 0,"x": 9,"y": 11,"type": "m"},
                {"created": 0,"x": 5,"y": 14,"type": "m"},{"created": 0,"x": 9,"y": 14,"type": "m"},{"created": 0,"x": 2,"y": 0,"type": "m"},
                {"created": 0,"x": 12,"y": 0,"type": "m"},{"created": 0,"x": 10,"y": 6,"type": "m"},{"created": 0,"x": 4,"y": 6,"type": "m"},
                {"created": 0,"x": 1,"y": 4,"type": "m"},{"created": 0,"x": 13,"y": 4,"type": "m"},{"created": 0,"x": 3,"y": 10,"type": "m"},
                {"created": 0,"x": 11,"y": 10,"type": "m"},{"created": 0,"x": 10,"y": 7,"type": "m"},{"created": 0,"x": 4,"y": 7,"type": "m"},
                {"created": 0,"x": 1,"y": 13,"type": "m"},{"created": 0,"x": 13,"y": 13,"type": "m"},{"created": 0,"x": 0,"y": 10,"type": "m"},
                {"created": 0,"x": 14,"y": 10,"type": "m"},{"created": 0,"x": 10,"y": 1,"type": "m"},{"created": 0,"x": 4,"y": 1,"type": "m"},
                {"created": 0,"x": 12,"y": 14,"type": "m"},{"created": 0,"x": 2,"y": 14,"type": "m"},
                {"created": 0,"x": 9,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 5,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 2,"y": 10,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 10,"type": "w","hp": 1},
                {"created": 0,"x": 2,"y": 7,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 7,"type": "w","hp": 1},
                {"created": 0,"x": 3,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 12,"y": 1,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 1,"type": "w","hp": 1},
                {"created": 0,"x": 4,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 14,"type": "w","hp": 1},
                {"created": 0,"x": 4,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 9,"type": "w","hp": 1},
                {"created": 0,"x": 13,"y": 5,"type": "w","hp": 1},{"created": 0,"x": 1,"y": 5,"type": "w","hp": 1},
                {"created": 0,"x": 2,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 2,"type": "w","hp": 1},
                {"created": 0,"x": 12,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 13,"type": "w","hp": 1},
                {"created": 0,"x": 14,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 13,"type": "w","hp": 1},
                {"created": 0,"x": 2,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 1,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 13,"y": 12,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 2,"type": "w","hp": 1},
                {"created": 0,"x": 3,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 12,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 11,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 11,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 10,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 10,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 9,"type": "w","hp": 1},
                {"created": 0,"x": 14,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 14,"type": "w","hp": 1},
                {"created": 0,"x": 4,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 13,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 12,"type": "w","hp": 1},
                {"created": 0,"x": 0,"y": 0,"type": "w","hp": 1},{"created": 0,"x": 14,"y": 0,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 5,"y": 7,"type": "w","hp": 1},{"created": 0,"x": 9,"y": 7,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 8,"type": "w","hp": 1},
                {"created": 0,"x": 11,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 3,"y": 9,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 3,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 3,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 4,"type": "o","hp": 3},{"created": 0,"x": 6,"y": 4,"type": "o","hp": 3},
                {"created": 0,"x": 10,"y": 4,"type": "o","hp": 3},{"created": 0,"x": 4,"y": 4,"type": "o","hp": 3},
                {"created": 0,"x": 13,"y": 8,"type": "o","hp": 3},{"created": 0,"x": 1,"y": 8,"type": "o","hp": 3},
                {"created": 0,"x": 2,"y": 5,"type": "o","hp": 3},{"created": 0,"x": 12,"y": 5,"type": "o","hp": 3},
                {"created": 0,"x": 10,"y": 10,"type": "o","hp": 3},{"created": 0,"x": 4,"y": 10,"type": "o","hp": 3},
                {"created": 0,"x": 8,"y": 2,"type": "o","hp": 3},{"created": 0,"x": 6,"y": 2,"type": "o","hp": 3},
                {"created": 0,"x": 6,"y": 3,"type": "o","hp": 3},{"created": 0,"x": 8,"y": 3,"type": "o","hp": 3}
            ],
            "world": {"width": 15,"height": 15},
            "tick": 0,
            "config": {"tick_rate_hz": 10,"game_duration_ticks": 300,"fire_spawn_interval_ticks": 2}
        },

        "training_arena_1" : {
            "game_id": "dev",
            "agents": {
                "a": {"agent_id": "a","unit_ids": ["c"]},   #,"e","g"
                "b": {"agent_id": "b","unit_ids": ["d"]}},  #,"f","h"
            "unit_state": {
                "c": {"coordinates": [2,3],"hp": 1,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
                "d": {"coordinates": [7,7],"hp": 1,"inventory": {"bombs": 0},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0},
                #"e": {"coordinates": [5,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "e","agent_id": "a","invulnerability": 0},
                #"f": {"coordinates": [9,12],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "f","agent_id": "b","invulnerability": 0},
                #"g": {"coordinates": [1,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "g","agent_id": "a","invulnerability": 0},
                #"h": {"coordinates": [13,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "h","agent_id": "b","invulnerability": 0}
            },
            "entities": [
                {"created": 0,"x": 6,"y": 7,"type": "m"},
                {"created": 0,"x": 8,"y": 7,"type": "m"},
                {"created": 0,"x": 7,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 7,"y": 8,"type": "w","hp": 1}
            ],
            "world": {"width": 15,"height": 15},
            "tick": 0,
            "config": {"tick_rate_hz": 100,"game_duration_ticks": 100,"fire_spawn_interval_ticks": 1}
        },

        "test_state": {
            "agents": {
                "a": {"agent_id": "a","unit_ids": ["c","e","g"]},
                "b": {"agent_id": "b","unit_ids": ["d","f","h"]}},
            "unit_state": {
                "c": {"coordinates": [4,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "c","agent_id": "a","invulnerability": 0},
                "d": {"coordinates": [10,10],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "d","agent_id": "b","invulnerability": 0},
                "e": {"coordinates": [3,0],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "e","agent_id": "a","invulnerability": 0},
                "f": {"coordinates": [11,0],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "f","agent_id": "b","invulnerability": 0},
                "g": {"coordinates": [10,3],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "g","agent_id": "a","invulnerability": 0},
                "h": {"coordinates": [4,3],"hp": 3,"inventory": {"bombs": 3},"blast_diameter": 3,"unit_id": "h","agent_id": "b","invulnerability": 0}},
            "entities": [
                {"created": 0,"x": 0,"y": 2,"type": "m"},{"created": 0,"x": 14,"y": 2,"type": "m"},
                {"created": 0,"x": 8,"y": 12,"type": "m"},{"created": 0,"x": 6,"y": 12,"type": "m"},
                {"created": 0,"x": 13,"y": 14,"type": "m"},{"created": 0,"x": 1,"y": 14,"type": "m"},
                {"created": 0,"x": 6,"y": 5,"type": "m"},{"created": 0,"x": 8,"y": 5,"type": "m"},
                {"created": 0,"x": 2,"y": 5,"type": "m"},{"created": 0,"x": 12,"y": 5,"type": "m"},
                {"created": 0,"x": 12,"y": 7,"type": "m"},{"created": 0,"x": 2,"y": 7,"type": "m"},
                {"created": 0,"x": 12,"y": 8,"type": "m"},{"created": 0,"x": 2,"y": 8,"type": "m"},
                {"created": 0,"x": 8,"y": 11,"type": "m"},{"created": 0,"x": 6,"y": 11,"type": "m"},
                {"created": 0,"x": 0,"y": 7,"type": "m"},{"created": 0,"x": 14,"y": 7,"type": "m"},
                {"created": 0,"x": 10,"y": 4,"type": "m"},{"created": 0,"x": 4,"y": 4,"type": "m"},
                {"created": 0,"x": 0,"y": 6,"type": "m"},{"created": 0,"x": 14,"y": 6,"type": "m"},
                {"created": 0,"x": 3,"y": 8,"type": "m"},{"created": 0,"x": 11,"y": 8,"type": "m"},
                {"created": 0,"x": 11,"y": 7,"type": "m"},{"created": 0,"x": 3,"y": 7,"type": "m"},
                {"created": 0,"x": 3,"y": 3,"type": "m"},{"created": 0,"x": 11,"y": 3,"type": "m"},
                {"created": 0,"x": 13,"y": 9,"type": "m"},{"created": 0,"x": 1,"y": 9,"type": "m"},
                {"created": 0,"x": 14,"y": 1,"type": "m"},{"created": 0,"x": 0,"y": 1,"type": "m"},
                {"created": 0,"x": 10,"y": 12,"type": "m"},{"created": 0,"x": 4,"y": 12,"type": "m"},
                {"created": 0,"x": 11,"y": 6,"type": "m"},{"created": 0,"x": 3,"y": 6,"type": "m"},
                {"created": 0,"x": 0,"y": 9,"type": "m"},{"created": 0,"x": 14,"y": 9,"type": "m"},
                {"created": 0,"x": 9,"y": 7,"type": "m"},{"created": 0,"x": 5,"y": 7,"type": "m"},
                {"created": 0,"x": 14,"y": 5,"type": "m"},{"created": 0,"x": 0,"y": 5,"type": "m"},
                {"created": 0,"x": 9,"y": 10,"type": "m"},{"created": 0,"x": 5,"y": 10,"type": "m"},
                {"created": 0,"x": 9,"y": 4,"type": "m"},{"created": 0,"x": 5,"y": 4,"type": "m"},
                {"created": 0,"x": 3,"y": 13,"type": "m"},{"created": 0,"x": 11,"y": 13,"type": "m"},
                {"created": 0,"x": 2,"y": 4,"type": "m"},{"created": 0,"x": 12,"y": 4,"type": "m"},
                {"created": 0,"x": 14,"y": 0,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 0,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 11,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 11,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 14,"type": "w","hp": 1},
                {"created": 0,"x": 1,"y": 1,"type": "w","hp": 1},{"created": 0,"x": 13,"y": 1,"type": "w","hp": 1},
                {"created": 0,"x": 3,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 9,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 5,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 5,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 13,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 13,"type": "w","hp": 1},
                {"created": 0,"x": 14,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 14,"type": "w","hp": 1},
                {"created": 0,"x": 5,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 9,"y": 8,"type": "w","hp": 1},
                {"created": 0,"x": 12,"y": 9,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 9,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 6,"y": 8,"type": "w","hp": 1},
                {"created": 0,"x": 3,"y": 5,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 5,"type": "w","hp": 1},
                {"created": 0,"x": 13,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 1,"y": 12,"type": "w","hp": 1},
                {"created": 0,"x": 0,"y": 11,"type": "w","hp": 1},{"created": 0,"x": 14,"y": 11,"type": "w","hp": 1},
                {"created": 0,"x": 13,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 1,"y": 8,"type": "w","hp": 1},
                {"created": 0,"x": 3,"y": 14,"type": "w","hp": 1},{"created": 0,"x": 11,"y": 14,"type": "w","hp": 1},
                {"created": 0,"x": 12,"y": 12,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 12,"type": "w","hp": 1},
                {"created": 0,"x": 2,"y": 3,"type": "w","hp": 1},{"created": 0,"x": 12,"y": 3,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 4,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 4,"type": "w","hp": 1},
                {"created": 0,"x": 9,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 5,"y": 2,"type": "w","hp": 1},
                {"created": 0,"x": 6,"y": 2,"type": "w","hp": 1},{"created": 0,"x": 8,"y": 2,"type": "w","hp": 1},
                {"created": 0,"x": 10,"y": 1,"type": "w","hp": 1},{"created": 0,"x": 4,"y": 1,"type": "w","hp": 1},
                {"created": 0,"x": 9,"y": 0,"type": "w","hp": 1},{"created": 0,"x": 5,"y": 0,"type": "w","hp": 1},
                {"created": 0,"x": 12,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 2,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 14,"y": 4,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 4,"type": "w","hp": 1},
                {"created": 0,"x": 4,"y": 6,"type": "w","hp": 1},{"created": 0,"x": 10,"y": 6,"type": "w","hp": 1},
                {"created": 0,"x": 14,"y": 8,"type": "w","hp": 1},{"created": 0,"x": 0,"y": 8,"type": "w","hp": 1},
                {"created": 0,"x": 8,"y": 10,"type": "o","hp": 3},{"created": 0,"x": 6,"y": 10,"type": "o","hp": 3},
                {"created": 0,"x": 5,"y": 1,"type": "o","hp": 3},{"created": 0,"x": 9,"y": 1,"type": "o","hp": 3},
                {"created": 0,"x": 6,"y": 3,"type": "o","hp": 3},{"created": 0,"x": 8,"y": 3,"type": "o","hp": 3},
                {"created": 0,"x": 3,"y": 4,"type": "o","hp": 3},{"created": 0,"x": 11,"y": 4,"type": "o","hp": 3},
                {"created": 0,"x": 9,"y": 6,"type": "o","hp": 3},{"created": 0,"x": 5,"y": 6,"type": "o","hp": 3},
                {"created": 0,"x": 6,"y": 0,"type": "o","hp": 3},{"created": 0,"x": 8,"y": 0,"type": "o","hp": 3},
                {"created": 0,"x": 5,"y": 11,"type": "o","hp": 3},{"created": 0,"x": 9,"y": 11,"type": "o","hp": 3}
            ],
            "world": {"width": 15,"height": 15},
            "tick": 0,
            "config": {"tick_rate_hz": 10,"game_duration_ticks": 300,"fire_spawn_interval_ticks": 2}
        }

    }

    if name is not None:
        return arena_templates.get(name)
    return arena_templates