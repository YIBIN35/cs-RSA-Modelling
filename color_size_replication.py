# Python doesn't natively support probabilistic programming - needs importing
# Vanilla model (Boolean Semantics)

import math

# Set Parameters
alpha = 1
costWeight = 0
c = 0

# Set States (objects)
states = ["big_blue", "big_red", "small_blue"]

# Set Utterances
utterances = ["big", "small", "blue", "red", "big_blue", "big_red", "small_blue"]

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

# Checks for utterance appearance inside state
def meaning(utterance, state):
    return utterance in state

# Cost function (in color-size experiment, manually set to 0)
cost = {
    "big": 0,
    "small":0,
    "blue": 0,
    "red": 0,
    "big_blue":c,
    "small_blue":c,
    "big_red":c
}

# Literal Listener function
def literal_listener(utterance):
    probabilities = {}
    total = 0
    for s in states:
        if meaning(utterance, s):
            probabilities[s] = 1 # Assign truth values
            total += 1 # Does not count in the cases for "0" as semantic value
        else:
            probabilities[s] = 0
            total += 0
    for s in probabilities:
        # Skips the step of multiplying prior because it does not mathematically affect the result
        # Divide the specific state with the sum of all states
        probabilities[s] /= total
    return probabilities

# Pragmatic Speaker function
def pragmatic_speaker(state):
    utterance_probs = {}
    total = 0.0
    for utt in utterances:
        # Retrieve the literal listener probability
        literal_listener_probs = literal_listener(utt)
        utterance_prob = literal_listener_probs.get(state)
        # For Boolean semantics, the only values are 1/0, check 1 for true
        if utterance_prob > 0:
            utility = alpha*math.log(utterance_prob)
        # Assign 0 (false states) negative infinity for calculation purposes
        else:
            utility = float('-inf')
        # The cost here is set to 0
        utterance_probs[utt] = math.exp(utility) - costWeight*cost[utt]
        total += utterance_probs[utt]
    # Normalize the probabilities
    for utterance in utterance_probs:
        utterance_probs[utterance] /= total
    return utterance_probs


# Test the functions
if __name__ == "__main__":
    print("Literal Listener Test")
    utterances = ["big", "small", "blue", "red", "big_blue", "big_red", "small_blue"]
    for u in utterances:
        results = literal_listener(u)
        print(f"\nUtterance: '{u}'")
        for state, prob in results.items():
            print(f"    P({state} | '{u}') = {prob:.2f}")

    print("Pragmatic Speaker Test")
    states = ["big_blue", "big_red", "small_blue"]
    for s in states:
        results = pragmatic_speaker(s)
        print(f"\nState: '{s}'")
        for utterance, prob in results.items():
            print(f"    P({utterance} | '{s}') = {prob:.2f}")




