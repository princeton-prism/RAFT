from copy import deepcopy

"""
Maps task type to a list of prompt groups.
Each prompt group contains a list of related prompt templates.
The purpose of prompt groups is to allow uniform sampling of prompts without sampling too many similar prompts.
So, you should uniformly sample a prompt group and then uniformly sample within the group.
"""
DEFAULT_PROMPT_TEMPLATES = {
    "pick_and_place": [
        # Formal "pick up and place" variations
        [
            "Pick up the {pickup_name} and place it in or on the {place_name}.",
            "Pick up the {pickup_name} and place it in the {place_name}.",
            "Pick up the {pickup_name} and place it on the {place_name}.",
        ],
        # Casual "put" variations
        [
            "Put the {pickup_name} on the {place_name}.",
            "Put the {pickup_name} in the {place_name}.",
            "Put the {pickup_name} in or on the {place_name}.",
        ],
        # Simple "move" command
        [
            "Move the {pickup_name} to the {place_name}.",
            "Move the {pickup_name} onto the {place_name}.",
            "Move the {pickup_name} into the {place_name}.",
        ],
        # "Grab" variations
        [
            "Grab the {pickup_name} and put it on the {place_name}.",
            "Grab the {pickup_name} and place it in the {place_name}.",
            "Grab the {pickup_name} and drop it on the {place_name}.",
        ],
        # "Take" variations
        [
            "Take the {pickup_name} and set it on the {place_name}.",
            "Take the {pickup_name} to the {place_name}.",
            "Take the {pickup_name} and put it in the {place_name}.",
        ],
        # Transfer/relocate
        [
            "Transfer the {pickup_name} to the {place_name}.",
            "Relocate the {pickup_name} to the {place_name}.",
        ],
        # Direct placement commands
        [
            "Place the {pickup_name} on the {place_name}.",
            "Place the {pickup_name} in the {place_name}.",
            "Place the {pickup_name} inside the {place_name}.",
        ],
        # Request-style commands
        [
            "Can you move the {pickup_name} to the {place_name}?",
            "Could you put the {pickup_name} on the {place_name}?",
            "Please pick up the {pickup_name} and place it on the {place_name}.",
        ],
        # Bring/carry variations
        [
            "Bring the {pickup_name} to the {place_name}.",
            "Carry the {pickup_name} over to the {place_name}.",
            "Bring the {pickup_name} over to the {place_name}.",
        ],
        # Get/fetch variations
        [
            "Get the {pickup_name} and put it on the {place_name}.",
            "Fetch the {pickup_name} and place it in the {place_name}.",
            "Get the {pickup_name} and set it on the {place_name}.",
        ],
        # Imperative short forms
        [
            "{pickup_name} to the {place_name}.",
            "{pickup_name} on the {place_name}.",
            "{pickup_name} goes on the {place_name}.",
        ],
        # Deposit/set down variations
        [
            "Set the {pickup_name} on the {place_name}.",
            "Set the {pickup_name} down on the {place_name}.",
            "Deposit the {pickup_name} in the {place_name}.",
        ],
        # Drop variations
        [
            "Drop the {pickup_name} on the {place_name}.",
            "Drop the {pickup_name} in the {place_name}.",
            "Drop the {pickup_name} into the {place_name}.",
        ],
    ],
    "pick_and_place_next_to": [
        # "Pick up and place" variations
        [
            "Pick up the {pickup_name} and place it next to the {place_name}.",
            "Pick up the {pickup_name} and place it near the {place_name}.",
        ],
        # Casual "put" variations
        [
            "Put the {pickup_name} next to the {place_name}.",
            "Put the {pickup_name} near the {place_name}.",
        ],
        # Simple "move" command
        [
            "Move the {pickup_name} next to the {place_name}.",
            "Move the {pickup_name} near the {place_name}.",
        ],
        # "Grab" variations
        [
            "Grab the {pickup_name} and put it next to the {place_name}.",
            "Grab the {pickup_name} and put it near the {place_name}.",
            "Grab the {pickup_name} and place it next to the {place_name}.",
            "Grab the {pickup_name} and place it near the {place_name}.",
            "Grab the {pickup_name} and drop it next to the {place_name}.",
            "Grab the {pickup_name} and drop it near the {place_name}.",
        ],
        # "Take" variations
        [
            "Take the {pickup_name} and set it next to the {place_name}.",
            "Take the {pickup_name} and set it near the {place_name}.",
            "Take the {pickup_name} and put it next to the {place_name}.",
            "Take the {pickup_name} and put it near the {place_name}.",
        ],
        # Transfer/relocate
        [
            "Transfer the {pickup_name} to be next to the {place_name}.",
            "Transfer the {pickup_name} to be near the {place_name}.",
            "Relocate the {pickup_name} to be next to the {place_name}.",
            "Relocate the {pickup_name} to be near the {place_name}.",
        ],
        # Direct placement commands
        [
            "Place the {pickup_name} next to the {place_name}.",
            "Place the {pickup_name} near the {place_name}.",
        ],
        # Request-style commands
        [
            "Can you move the {pickup_name} to be next to the {place_name}?",
            "Can you move the {pickup_name} to be near the {place_name}?",
            "Could you put the {pickup_name} next to the {place_name}?",
            "Could you put the {pickup_name} near the {place_name}?",
            "Please pick up the {pickup_name} and place it next to the {place_name}.",
            "Please pick up the {pickup_name} and place it near the {place_name}.",
        ],
        # Get/fetch variations
        [
            "Get the {pickup_name} and put it next to the {place_name}.",
            "Fetch the {pickup_name} and place it near the {place_name}.",
        ],
        # Drop variations
        [
            "Drop the {pickup_name} next to the {place_name}.",
            "Drop the {pickup_name} near the {place_name}.",
        ],
    ],
    "pick": [
        [
            "Pick up the {pickup_obj_name}.",
            "Lift the {pickup_obj_name}.",
            "Pick the {pickup_obj_name}.",
            "Grab the {pickup_obj_name}.",
            "Get the {pickup_obj_name}.",
        ]
    ],
    # OpeningTask (FrankaOpenDataGenConfig) emits task_type="open" with a single
    # referral key, pickup_obj_name — same shape as "pick".
    "open": [
        # Direct commands
        [
            "Open the {pickup_obj_name}.",
            "Open up the {pickup_obj_name}.",
            "Pull open the {pickup_obj_name}.",
        ],
        # Request-style
        [
            "Can you open the {pickup_obj_name}?",
            "Could you open the {pickup_obj_name}?",
            "Please open the {pickup_obj_name}.",
        ],
    ],
}

DEFAULT_PROMPT_TEMPLATES["pick_and_place_color"] = deepcopy(DEFAULT_PROMPT_TEMPLATES["pick_and_place"])
