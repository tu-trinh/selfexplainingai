from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
IDX_TO_STATE = {
    0: "open",
    1: "closed",
    2: "locked"
}
IDX_TO_DIR = {
    0: "east",
    1: "south",
    2: "west",
    3: "north"
}
ACTION_TO_IDX = {
    "left": 0,
    "right": 1, 
    "forward": 2,
    "pickup": 3, 
    "putdown": 4,
    "unlock": 5
}

def get_unit_desc(cell):
    obj = IDX_TO_OBJECT[cell[0]]
    color = IDX_TO_COLOR[cell[1]]
    if obj == "door":
        state = IDX_TO_STATE[cell[2]]
        beginning = "an" if state == "open" else "a"
        desc = beginning + " " + state + " " + color + " " + obj
    elif obj == "empty":
        desc = "walkable floor"
    elif obj == "unseen":
        desc = ""
    else:
        desc = "a " + color + " " + obj
    return desc

def get_obs_desc(obs, agent_view_size = 3, difficulty = 3):
    img = obs["image"]
    if difficulty == 3:
        top_left = get_unit_desc(img[0][0])
        top_middle = get_unit_desc(img[1][0])
        top_right = get_unit_desc(img[2][0])
        middle_left = get_unit_desc(img[0][1])
        center = get_unit_desc(img[1][1])
        middle_right = get_unit_desc(img[2][1])
        direct_left = get_unit_desc(img[0][2])
        agent_pos = get_unit_desc(img[1][2]) # img[1][2] is the agent position
        direct_right = get_unit_desc(img[2][2])
        agent_desc = ""
        if agent_pos != "" and "floor" not in agent_pos:
            agent_desc = f"Finally, you are holding {agent_pos}."
        description = f"You are facing {IDX_TO_DIR[obs['direction']]}. Your field of vision is a {agent_view_size}-by-{agent_view_size} square in which you are located at the bottom middle. Directly to your left is {direct_left} and directly to your right is {direct_right}. Right in front of you is {center}. In front but to the left you see {middle_left}. In front but to the right you see {middle_right}. Farther away, from left to right, you see {top_left}, {top_middle}, and {top_right}. {agent_desc}"
    elif difficulty == 5:
        objects = []
        floor_locs = set()
        for i in range(5):
            for j in range(5):
                # img[2][4] is the agent position
                if not (i, j) in [(2, 4), (1, 4), (3, 4), (2, 3)]:
                    obj = get_unit_desc(img[i][j])
                    if obj != "":
                        objects.append(obj)
                        if obj == "walkable floor":
                            if i == 2:
                                floor_locs.add("in front")
                            elif j == 4:
                                if i < 2:
                                    floor_locs.add("to the left")
                                else:
                                    floor_locs.add("to the right")
        direct_next = [get_unit_desc(img[1][4]), get_unit_desc(img[3][4]), get_unit_desc(img[2][3])]
        agent_pos = get_unit_desc(img[2][4])
        desc_set = {}
        processed_set = set()
        for obj in objects:
            if obj in desc_set and obj not in processed_set:
                desc_set[obj] = " ".join(desc_set[obj].split()[1:]) + "s"
                processed_set.add(obj)
            elif obj not in desc_set:
                if obj != "walkable floor":
                    desc_set[obj] = obj
        obj_descs = list(desc_set.values())
        floor_locs = list(floor_locs)
        if len(floor_locs) > 0:
            if len(floor_locs) > 1:
                floor_desc = f"There is walkable floor {', '.join(floor_locs[:-1])}, and {floor_locs[-1]} of you."
            else:
                floor_desc = f"There is walkable floor {floor_locs[0]} of you."
        else:
            floor_desc = ""
        agent_desc = ""
        if agent_pos != "" and "floor" not in agent_pos:
            agent_desc = f"Finally, you are holding {agent_pos}."
        if len(obj_descs) > 1:
            description = f"You are facing {IDX_TO_DIR[obs['direction']]}. Your field of vision is a 5x5 square in which you are located at the bottom middle. Directly to your left, you see {direct_next[0]}. Directly to your right, you see {direct_next[1]}. Directly in front of you is {direct_next[2]}. Elsewhere around you, you see {', '.join(obj_descs[:-1])}, and {obj_descs[-1]}. {floor_desc} {agent_desc}"
        else:
            description = f"You are facing {IDX_TO_DIR[obs['direction']]}. Your field of vision is a 5x5 square in which you are located at the bottom middle. Directly to your left, you see {direct_next[0]}. Directly to your right, you see {direct_next[1]}. Directly in front of you is {direct_next[2]}. Elsewhere around you, you see {obj_descs[0]}. {floor_desc} {agent_desc}"
    return description
