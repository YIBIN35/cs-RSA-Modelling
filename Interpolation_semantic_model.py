# Interpolation Semantic Model

import math

# Set Parameters
# Here alpha is equivalent to beta in the paper
# Set beta_fixed as MAP value in Degen paper
alpha = 13.7
beta_fixed = 0.69
costWeight = 0 

# Fixed sem values
size_semvalue = 0.8
state_semvalue = 0.9
nominal_semvalue = 0.99

# Empirical sem values
# nominal_typical_semvalue = 0.9
# nominal_atypical_semvalue = 0.35
# nominal_specified_typical_semvalue = 0.97
# nominal_specified_typical_semvalue = 0.66

# Set states (of objects) and utterances
world = {
    "singleton_marked":  [{"size": "None", "state": "open", "nominal": "door"},
                          {"size": "None", "state": "None" , "nominal": "other1"},
                          {"size": "None", "state": "None" , "nominal": "other2"},
                          {"size": "None", "state": "None" , "nominal": "other3"}],
    "singleton_unmarked":[{"size": "None", "state": "closed", "nominal": "door"},
                          {"size": "None", "state": "None" , "nominal": "other1"},
                          {"size": "None", "state": "None" , "nominal": "other2"},
                          {"size": "None", "state": "None" , "nominal": "other3"}],

    "pair_marked":       [{"size": "big", "state": "open", "nominal": "door"},
                          {"size": "small", "state": "open" , "nominal": "door"},
                          {"size": "None", "state": "None" , "nominal": "other2"},
                          {"size": "None", "state": "None" , "nominal": "other3"}],
    "pair_unmarked":     [{"size": "big", "state": "closed", "nominal": "door"},
                          {"size": "small", "state": "closed" , "nominal": "door"},
                          {"size": "None", "state": "None" , "nominal": "other2"},
                          {"size": "None", "state": "None" , "nominal": "other3"}]
}


# Separately define color & size for further calculations
sizes = ["big", "small"]
states = ["open", "closed"]
nominals = ["door", "other1", "other2", "other3"]

# Meaning function (Using continuous semantics instead of Boolean)
def meaning(utt, obj, print_value=False):
    split_words = utt.split("_")

    size_val = 1
    state_val = 1
    nominal_val = 1

    # fixed compositional semantic value function
    for word in split_words:
        if word in sizes:
            if word == obj['size']:
                size_val = size_semvalue
            elif obj['size'] == "None":
                size_val = size_semvalue
            else:
                size_val = 1- size_semvalue
        if word in states:
            if word == obj['state']:
                state_val = state_semvalue
            elif obj['state'] == "None":
                state_val = state_semvalue
            else:
                state_val = 1- state_semvalue
        if word in nominals:
            if word == obj['nominal']:
                nominal_val = nominal_semvalue
            elif obj['state'] == "None":
                nominal_val = nominal_semvalue
            else:
                nominal_val = 1- nominal_semvalue

    fixed_sem_value = size_val * state_val * nominal_val


    # Empirical semantic value function
    # parse utterance
    size_words = []
    state_words = []
    nominal_words = []
    for word in split_words:
        if word in sizes:
            size_words.append(word)
        elif word in states:
            state_words.append(word)
        elif word in nominals:
            nominal_words.append(word)
        else:
            raise Exception("Something went wrong")

    empirical_sem_value = 0
    for word in nominal_words:
        if word == obj["nominal"]:
            # check table 7 to understand the code here. right now everything is hardcoded!!!
            if len(state_words) == 0:
                if obj['state'] == 'closed':
                    empirical_sem_value = 0.98
                else:
                    empirical_sem_value = 0.66

            elif state_words[0] == 'closed':
                if obj["state"] == 'closed':
                    empirical_sem_value = 0.97
                else:
                    empirical_sem_value = 0.30

            elif state_words[0] == 'open':
                if obj["state"] == 'closed':
                    empirical_sem_value = 0.22
                else:
                    empirical_sem_value = 0.91
        else:
            empirical_sem_value = 1 - nominal_semvalue

    if print_value == True:
        print("utterance", size_words, state_words, nominal_words)
        print("object", obj)
        print(beta_fixed, size_val, state_val, nominal_val, 1-beta_fixed, empirical_sem_value)

    # Add two models together
    sem_value = (1 - beta_fixed) * empirical_sem_value + beta_fixed * fixed_sem_value
    if sem_value != 0:
        return sem_value

# Literal Listener function (math.exp applies to raw semantic value)
def literal_listener(utterance, world):
    probabilities = {}
    total = 0
    for obj in world:
        # Assign continuous semantic values instead of 1/0
        sem_val = meaning(utterance, obj)
        item = tuple(obj.items())
        # Map each semantic value to an item of (size, color)
        # probabilities[item] = math.exp(sem_val)
        probabilities[item] = math.exp(1.34*sem_val)
        total += math.exp(sem_val)
    # Normalize the semantic values
    for item in probabilities:
        probabilities[item] /= total
    return probabilities

# No cost for this model
# def cost(utt):
#     return len(utt.split("_"))

def cost(utt):
    return len(utt.split("_"))

# Pragmatic Speaker function
def pragmatic_speaker(obj, world, utterances):
    # Transform the object from dictionary tuple for .get() function afterwards
    obj_key = tuple(obj.items())
    utterance_probs = {}
    total = 0.0
    for utt in utterances:
        # Retrieve the corresponding literal listener values for each utterance
        literal_listener_prob = literal_listener(utt, world)
        # Match the utterance with the state
        utterance_prob = literal_listener_prob.get(obj_key)
        # Apply the pragmatic speaker function
        utility = alpha * math.log(utterance_prob) - costWeight * cost(utt)
        utterance_probs[utt] = math.exp(utility)
        total += math.exp(utility)
    # Normalize the values
    for item in utterance_probs:
        utterance_probs[item] /= total
    return utterance_probs


# def overspecification_rate(results):
#     # right now this is just a hack
#     result_list = list(results.values())
#     modification = sum(result_list[1:3]) / sum(result_list[:3])
#     return modification


# Test the functions
if __name__ == "__main__":
    # The last row in this literal listener table has "0.28, 0,23, 0.50" due to decimal rounding
    # The values should add up to 1 in real calculation

    print(meaning('closed_door', {"size": "None", "state": "open", "nominal": "door"}, print_value=True))

    # print("Literal Listener Test")
    # for u in utterances:
    #     results = literal_listener(u, this_world)
    #     print(f"\nUtterance: '{u}'")
    #     for state, prob in results.items():
    #         print(f"    P({state} | '{u}') = {prob:.2f}")


    # print("Pragmatic Speaker Test")
    # for obj in this_world:
    #     results = pragmatic_speaker(obj, this_world)
    #     print(f"\nobj: '{obj}'")
    #     for utterance, prob in results.items():
    #         print(f"    P({utterance} | '{obj}') = {prob:.2f}")


    # for condition, this_world in world.items():
    #     results = pragmatic_speaker(this_world[0], this_world, utterances)
    #     print(f"\n{condition}, obj: '{this_world[0]}'")
    #     print(f"overspecification rate: {overspecification_rate(results)}")
    #     for utterance, prob in results.items():
    #         print(f"    P({utterance} | '{this_world[0]}') = {prob:.2f}")


    for condition in ["singleton_marked", "singleton_unmarked"]:
        this_world = world[condition]
        utterances = [
            "door",
            "open_door",
            "closed_door",
            "other1",
            "other2",
            "other3"
        ]
        results = pragmatic_speaker(this_world[0], this_world, utterances)
        overspecification_rate = sum(list(results.values())[1:3]) / sum(list(results.values())[:3])
        print(f"\n{condition}, obj: '{this_world[0]}'")
        print(f"overspecification rate: {overspecification_rate}")
        for utterance, prob in results.items():
            print(f"    P({utterance} | '{this_world[0]}') = {prob:.2f}")

    for condition in ["pair_marked", "pair_unmarked"]:
        this_world = world[condition]
        utterances = [
            "door",
            "big_door",
            "small_door",
            "open_door",
            "closed_door",
            "big_open_door",
            "small_open_door",
            "big_closed_door",
            "small_closed_door",
            "other1",
            "other2",
            "other3"
        ]
        results = pragmatic_speaker(this_world[0], this_world, utterances)
        overspecification_rate = sum(list(results.values())[3:9]) / sum(list(results.values())[:9])
        print(f"\n{condition}, obj: '{this_world[0]}'")
        print(f"overspecification rate: {overspecification_rate}")
        for utterance, prob in results.items():
            print(f"    P({utterance} | '{this_world[0]}') = {prob:.2f}")

