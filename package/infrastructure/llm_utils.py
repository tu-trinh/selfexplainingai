def convert_response_to_action(resp, additional_actions):
    """
    Helper function to parse LLM response when choosing an action
    """
    return resp

    # Only needed for automation
    # if "forward" in resp or "1" in resp:
    #     return 1
    # if "left" in resp or "2" in resp:
    #     return 2
    # if "right" in resp or "3" in resp:
    #     return 3
    # if "pick" in resp or "4" in resp:
    #     return 4
    # if "put" in resp or "5" in resp:
    #     return 5
    # if "unlock" in resp or "6" in resp:
    #     return 6
    # if "open" in resp or "7" in resp:
    #     return 7
    # if "close" in resp or "8" in resp:
    #     return 8
    # return None