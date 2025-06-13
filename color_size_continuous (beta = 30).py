# Continuous Semantics Model (beta = 30)

import math

# Set Parameters
# Here alpha is equivalent to beta in the paper
alpha = 30
costWeight = 1
size_semvalue = 0.8
color_semvalue = 0.99
size_cost = 0
color_cost = 0

# Set states (of objects) and utterances
states = [
    {"size": "big", "color": "blue"},
    {"size": "big", "color": "red"},
    {"size": "small", "color": "blue"}
]

utterances = ["big", "small", "blue", "red", "big_blue", "small_blue", "big_red"]

# Separately define color & size for further calculations
colors = ["red", "blue"]
sizes = ["big", "small"]

# Computation of Prior Probability (before any linguistic knowledge applies)
# Prior Probability of States/Objects (for literal listener function)
def state_prior():
    prior = {}
    total_states = len(states)
    for s in states:
        prior[s] = 1/total_states
    return prior

# Prior Probability of Utterances (for pragmatic speaker function)
def utterance_prior():
    prior = {}
    total_utterances = len(utterances)
    for u in utterances:
        prior[u] = 1/total_utterances
    return prior

# Meaning function (Using continuous semantics instead of Boolean)
def meaning(utt, state):
    split_words = utt.split("_")
    # Check whether the utterance is single or double modifiers
    if len(split_words) == 1:
        word = split_words[0]
        # Check if the word is color or size modifier
        if word in colors:
            # Check whether the utterance corresponds to the referent
            if word == state["color"]:
                return color_semvalue
            else:
                return 1 - color_semvalue
        if word in sizes:
            if word == state["size"]:
                return size_semvalue
            else:
                return 1 - size_semvalue
    # If the word contains two modifiers (assuming the order is SIZE_COLOR)
    elif len(split_words) == 2:
        size_word = split_words[0]
        color_word = split_words[1]
        size_val = 0
        color_val = 0
        if size_word in sizes:
            if size_word == state["size"]:
                size_val = size_semvalue
            else:
                size_val = 1 - size_semvalue
        if color_word in colors:
            if color_word == state["color"]:
                color_val = color_semvalue
            else:
                color_val = 1 - color_semvalue
        # Multiply the color and size semantic values
        sem_value = size_val * color_val
        # Check if both modifiers are size or color modifiers
        if sem_value != 0:
            return sem_value
        # Raise an error if the modifiers are not in the two categories
        else:
            raise ValueError(f"Bad utterance length: len{split_words}")


# Define costs (here it is still set to 0)
cost = {
    "big": size_cost,
    "small": size_cost,
    "blue": color_cost,
    "red": color_cost,
    "big_blue": size_cost + color_cost,
    "small_blue": size_cost + color_cost,
    "big_red": size_cost + color_cost
}

# Literal Listener function (math.exp applies to raw semantic value)
def literal_listener(utterance):
    probabilities = {}
    total = 0
    for obj in states:
        # Assign continuous semantic values instead of 1/0
        sem_val = meaning(utterance, obj)
        item = tuple(obj.items())
        # Map each semantic value to an item of (size, color)
        probabilities[item] = math.exp(sem_val)
        total += math.exp(sem_val)
    # Normalize the semantic values
    for item in probabilities:
        probabilities[item] /= total
    return probabilities

# Pragmatic Speaker function
def pragmatic_speaker(state):
    # Transform the object from dictionary tuple for .get() function afterwards
    state_key = tuple(state.items())
    utterance_probs = {}
    total = 0.0
    for utt in utterances:
        # Retrieve the corresponding literal listener values for each utterance
        literal_listener_prob = literal_listener(utt)
        # Match the utterance with the state
        utterance_prob = literal_listener_prob.get(state_key)
        # Apply the pragmatic speaker function
        utility = alpha * math.log(utterance_prob) - costWeight * cost[utt]
        utterance_probs[utt] = math.exp(utility)
        total += math.exp(utility)
    # Normalize the values
    for item in utterance_probs:
        utterance_probs[item] /= total
    return utterance_probs


# Test the functions
if __name__ == "__main__":
    # The last row in this literal listener table has "0.28, 0,23, 0.50" due to decimal rounding
    # The values should add up to 1 in real calculation
    print("Literal Listener Test")
    utterances = ["big", "small", "blue", "red", "big_blue", "big_red", "small_blue"]
    for u in utterances:
        results = literal_listener(u)
        print(f"\nUtterance: '{u}'")
        for state, prob in results.items():
            print(f"    P({state} | '{u}') = {prob:.2f}")

    print("Pragmatic Speaker Test")
    states = [
        {"size": "big", "color": "blue"},
        {"size": "big", "color": "red"},
        {"size": "small", "color": "blue"}
    ]
    for s in states:
        results = pragmatic_speaker(s)
        print(f"\nState: '{s}'")
        for utterance, prob in results.items():
            print(f"    P({utterance} | '{s}') = {prob:.2f}")

