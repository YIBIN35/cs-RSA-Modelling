# Continuous Semantics Model (beta = 1)

import math

# Set Parameters
# Here alpha is equivalent to beta in the paper
alpha = 12
costWeight = 6

size_semvalue = 0.8
state_semvalue = 0.9
nominal_semvalue = 0.99
nominal_typical_semvalue = 0.9
nominal_atypical_semvalue = 0.35

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
        "singleton_unmarked":[{"size": "None", "state": "closed", "nominal": "door"}, 
                              {"size": "None", "state": "None" , "nominal": "other1"}, 
                              {"size": "None", "state": "None" , "nominal": "other2"}, 
                              {"size": "None", "state": "None" , "nominal": "other3"}],
        # "pair_marked":       [{"size": "big", "state": "open", "nominal": "door"}, 
        #                       {"size": "small", "state": "open" , "nominal": "door"}, 
        #                       {"size": "None", "state": "None" , "nominal": "other2"}, 
        #                       {"size": "None", "state": "None" , "nominal": "other3"}],
        # "pair_unmarked":     [{"size": "big", "state": "closed", "nominal": "door"}, 
        #                       {"size": "small", "state": "closed" , "nominal": "door"}, 
        #                       {"size": "None", "state": "None" , "nominal": "other2"}, 
        #                       {"size": "None", "state": "None" , "nominal": "other3"}] 
        }

# utterances = ["big", "small", "blue", "red", "big_blue", "small_blue", "big_red"]
utterances = [
              "door",
              "open_door",
              "closed_door",
              # "big_door",
              # "small_door",
              # "big_open_door",
              # "small_open_door",
              # "big_closed_door",
              # "small_closed_door",
              "other1", 
              "other2", 
              "other3"
              ]

# Separately define color & size for further calculations
sizes = ["big", "small"]
states = ["open", "closed"]
nominals = ["door", "other1", "other2", "other3"]

# Meaning function (Using continuous semantics instead of Boolean)
def meaning(utt, obj, print_value=False):
    split_words = utt.split("_")

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

    # check utterance with the world state
    # size_value = 1
    # state_value = state_semvalue
    # nominal_value = nominal_semvalue

    # for word in size_words:
    #     if obj['size'] == 'None':
    #         size_value = size_semvalue
    #     elif word == obj["size"]:
    #         size_value = size_semvalue
    #     else:
    #         size_value = 1 - size_semvalue

    # for word in state_words:
    #     if obj['state'] == 'None':
    #         state_value = state_semvalue
    #     elif word == obj["state"]:
    #         state_value = state_semvalue
    #     else:
    #         state_value = 1 - state_semvalue

    for word in nominal_words:
        if word == obj["nominal"]:
            if len(state_words) == 0 and obj['state'] == 'closed':
                state_nominal_value = nominal_typical_semvalue
            elif len(state_words) == 0 and obj['state'] != 'closed':
                state_nominal_value = nominal_atypical_semvalue
            elif state_words[0] == obj["state"]: # right now assume that there is only one word in state_words
                state_nominal_value = nominal_semvalue
            elif state_words[0] != obj["state"]:
                state_nominal_value = 1 - nominal_semvalue
            else:
                raise Exception("Something went wrong")

        else:
            state_nominal_value = 1 - nominal_semvalue

    if print_value == True:
        print("utterance", size_words, state_words, nominal_words)
        print("object", obj)
        print(state_nominal_value)
    sem_value =  state_nominal_value 
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
        probabilities[item] = math.exp(sem_val)
        total += math.exp(sem_val)
    # Normalize the semantic values
    for item in probabilities:
        probabilities[item] /= total
    return probabilities

def cost(utt):
    return len(utt.split("_"))

# Pragmatic Speaker function
def pragmatic_speaker(obj, world):
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


def overspecification_rate(results):
    # right now this is just a hack
    result_list = list(results.values())
    modification = sum(result_list[1:3]) / sum(result_list[:3])
    return modification


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


    for condition, this_world in world.items():
        results = pragmatic_speaker(this_world[0], this_world)
        print(f"\n{condition}, obj: '{this_world[0]}'")
        print(f"overspecification rate: {overspecification_rate(results)}")
        for utterance, prob in results.items():
            print(f"    P({utterance} | '{this_world[0]}') = {prob:.2f}")
