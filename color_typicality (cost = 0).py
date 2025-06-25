# Reverse-engineering cs-RSA model for color-typicality - Trial: cost = 0
import math

# Set states
states = ["yellow_banana", "brown_banana", "blue_banana", "other"]

# Set utterances
utterances = ["banana", "yellow_banana", "brown_banana", "blue_banana", "other"]

# Directly assign semantic values for each utterance
# Each state (obj) has a dictionary of values for each utterance
meaning = {
    "yellow_banana": {"banana": 0.9, "yellow_banana": 0.99, "brown_banana": 0.01,
                      "blue_banana": 0.01, "other": 0.01},
    "brown_banana": {"banana": 0.35, "yellow_banana": 0.01, "brown_banana": 0.99,
                     "blue_banana": 0.01, "other": 0.01},
    "blue_banana": {"banana": 0.1, "yellow_banana": 0.01, "brown_banana": 0.01,
                    "blue_banana": 0.99, "other": 0.01},
    "other": {"banana": 0.01, "yellow_banana": 0.01, "brown_banana": 0.01,
              "blue_banana": 0.01, "other": 0.99}
}

# For this trial, set the cost parameters to 0
alpha = 12
ColorWeight = 0
TypeWeight = 0

# Compute the literal listener function
def literal_listener(utterance):
    probabilities = {}
    total = 0
    for s in states:
        # Directly search for the semantic value
        if s in meaning.keys():
            sem_value = meaning[s][utterance]
            # Perform the softmax() process
            probabilities[s] = math.exp(sem_value)
            total += math.exp(sem_value)
        else:
            # Raise error if state not found
            raise KeyError(f"State '{s}' not found in states")
    # Normalize the probabilities
    for item in probabilities:
        probabilities[item] /= total
    return probabilities

# Pragmatic Speaker function
def pragmatic_speaker(state):
    utterance_probs = {}
    total = 0.0
    for utt in utterances:
        # Retrieve the literal listener values for the corresponding utterance
        literal_listener_prob = literal_listener(utt)
        # Get the utterance's probability
        utterance_prob = literal_listener_prob.get(state)
        # Calculate utility, currently cost is 0 and alpha is set to 1
        utility = alpha * math.log(utterance_prob) - ColorWeight * 1 - TypeWeight * 1
        utterance_probs[utt] = math.exp(utility)
        total += math.exp(utility)
    # Normalize the values
    for item in utterance_probs:
        utterance_probs[item] /= total
    return utterance_probs

# Test the functions
if __name__ == "__main__":
    print("Testing literal listener function for all utterances:\n")
    for utt in utterances:
        print(f"Utterance: '{utt}'")
        probs = literal_listener(utt)
        for state, prob in probs.items():
            print(f"  {state}: {prob:.2f}")
        print()

    print("Testing pragmatic speaker function for all states:\n")
    for state in states:
        print(f"State: '{state}'")
        probs = pragmatic_speaker(state)
        for utt, prob in probs.items():
            print(f"  {utt}: {prob:.2f}")
        print()
