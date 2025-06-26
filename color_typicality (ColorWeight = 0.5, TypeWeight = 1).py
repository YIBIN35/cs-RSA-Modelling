# Reverse-engineering cs-RSA model for color-typicality - Trial: cost != 0
# ColorWeight = 1, TypeWeight = 0.5
import math

# Set states for each set
states_by_set = {
    "yellow_set": ["other", "yellow_banana", "other"],
    "brown_set": ["other", "brown_banana", "other"],
    "blue_set": ["other", "blue_banana", "other"]
}

# Set utterances for each set
utterances_by_set = {
    "yellow_set": ["banana", "yellow_banana", "other"],
    "brown_set": ["banana", "brown_banana", "other"],
    "blue_set": ["banana", "blue_banana", "other"]
}

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

# Set parameters
# Since cost is no longer zero, the weight for cost in the utility function needs to be set
alpha_informativeness = 12
alpha_cost = 5
ColorWeight = 0.5
TypeWeight = 1.0

# Compute the cost function
def cost(utterance):
    color_words = ["yellow", "brown", "blue", "other"]
    type_words = ["banana", "other"]
    color_mention = 0
    type_mention = 0

    words = utterance.split("_")
    # Check for number of appearances (1/0) in color and type words
    # Corresponds to the ._intersection() in the original model
    for word in words:
        if word in color_words:
            color_mention += 1
        if word in type_words:
            type_mention += 1
    word_cost = ColorWeight * color_mention + TypeWeight * type_mention
    return word_cost


# Compute the literal listener function
def literal_listener(utterance, states):
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
def pragmatic_speaker(state, states, utterances):
    utterance_probs = {}
    total = 0.0
    for utt in utterances:
        # Retrieve the literal listener values for the corresponding utterance
        literal_listener_prob = literal_listener(utt, states)
        # Get the utterance's probability (literal listener)
        utterance_prob = literal_listener_prob.get(state)
        # Calculate the cost
        utt_cost = cost(utt)
        # Calculate utility, currently cost is 0 and alpha is set to 1
        utility = alpha_informativeness * math.log(utterance_prob) - alpha_cost * utt_cost
        utterance_probs[utt] = math.exp(utility)
        total += math.exp(utility)
    # Normalize the values
    for item in utterance_probs:
        utterance_probs[item] /= total
    return utterance_probs

# Test the functions
if __name__ == "__main__":
    for state_set in states_by_set:
        print(f"\n===== Testing {state_set} =====\n")
        states = states_by_set[state_set]
        utterances = utterances_by_set[state_set]

        #     print("Testing literal listener function:\n")
        #     for utt in utterances:
        #         print(f"Utterance: '{utt}'")
        #         probs = literal_listener(utt, states)
        #         for state, prob in probs.items():
        #             print(f"  {state}: {prob:.2f}")
        #         print()

        print("Testing pragmatic speaker function:\n")
        for state in states:
            probs = pragmatic_speaker(state, states, utterances)
            if "banana" in state:
                print(f"State: '{state}'")
                for utt, prob in probs.items():
                    print(f"  {utt}: {prob:.2f}")
                print()
