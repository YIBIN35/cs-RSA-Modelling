import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing, minimize, brute
plt.ion()

parker_world = {
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

class cs_rsa:

    def __init__(
            self,
            world=None,
            alpha = 13.7,
            beta_fixed = 0.69,
            costWeight = 0 ,
            typicalityWeight = 1.34,
            size_semvalue = 0.8,
            state_semvalue_marked = 0.95,
            state_semvalue_unmarked = 0.9,
            nominal_semvalue = 0.99
            ):

        # parameters
        self.alpha = alpha
        self.beta_fixed = beta_fixed
        self.costWeight = costWeight
        self.typicalityWeight = typicalityWeight

        # semantic values
        self.size_semvalue = size_semvalue
        self.state_semvalue_marked = state_semvalue_marked
        self.state_semvalue_unmarked = state_semvalue_unmarked
        self.nominal_semvalue = nominal_semvalue

        # vocab
        self.sizes = ["big", "small"]
        self.states = ["open", "closed"]
        self.nominals = ["door", "other1", "other2", "other3"]

        # world
        self.world = world

    def meaning(self, utt, obj, print_value=False):
        split_words = utt.split("_")

        # if size/state/nominal_word=None: sem_val=1
        size_val = 1
        state_val = 1
        nominal_val = 1

        # fixed/compositional semantic value function
        for word in split_words:
            if word in self.sizes:
                if word == obj['size']:
                    size_val = self.size_semvalue  # big -> big | small -> small
                else:
                    if obj['size'] != "None":
                        size_val = 1 - self.size_semvalue  # small -> big | big -> small
                    else:
                        size_val = self.size_semvalue # big -> None | small -> None
            if word in self.states: 
                if word == 'open':
                    if obj['state'] == 'open':
                        state_val = self.state_semvalue_marked # open -> open
                    else:
                        state_val = 1 - self.state_semvalue_marked # open -> None | open -> closed
                else: # word == 'closed':
                    if obj['state'] == 'closed':
                        state_val = self.state_semvalue_unmarked # closed -> closed
                    else:
                        state_val = 1 - self.state_semvalue_unmarked # closed -> open | closed -> None
            if word in self.nominals:
                if obj['nominal'] != "None":
                    if word == obj['nominal']:
                        nominal_val = self.nominal_semvalue # door -> door
                    else:
                        nominal_val = 1- self.nominal_semvalue # door -> other
                else:
                    raise Exception("Something went wrong") # door -> None | other -> None

        fixed_sem_value = size_val * state_val * nominal_val

        # Empirical/non-compositional semantic value function
        size_words = []
        state_words = []
        nominal_words = []
        for word in split_words:
            if word in self.sizes:
                size_words.append(word)
            elif word in self.states:
                state_words.append(word)
            elif word in self.nominals:
                nominal_words.append(word)
            else:
                raise Exception("Something went wrong")

        # check table 7 to understand the code here. right now everything is hardcoded!!!
        empirical_sem_value = 0
        for word in nominal_words:
            if word == obj["nominal"]:
                if len(state_words) == 0:
                    if obj['state'] == 'closed':
                        empirical_sem_value = 0.98 # door -> closed door
                    else:
                        empirical_sem_value = 0.66 # door -> open door

                elif state_words[0] == 'closed':
                    if obj["state"] == 'closed':
                        empirical_sem_value = 0.97 # closed_door -> closed door
                    else:
                        empirical_sem_value = 0.30 # closed_door -> open door

                elif state_words[0] == 'open':
                    if obj["state"] == 'closed':
                        empirical_sem_value = 0.22 # open_door -> closed door
                    else:
                        empirical_sem_value = 0.91 # open_door -> open door
            else:
                empirical_sem_value = 1 - self.nominal_semvalue # door -> other

        # Add two models together
        sem_value = (1 - self.beta_fixed) * empirical_sem_value + self.beta_fixed * fixed_sem_value

        if print_value == True:
            print("utterance:", f"size:{size_words}, state:{state_words}, nominal:{nominal_words}")
            print("object:", obj)
            print(f"sem_val: beta_fixed:{self.beta_fixed}, size:{size_val}, state:{state_val}, nominal:{nominal_val}\n         1-beta_fixed:{1-self.beta_fixed}, emp:{empirical_sem_value}")
            print(f"sem_val: {sem_value}")
            print("")

        return sem_value

    def literal_listener(self, utterance):
        probabilities = {}
        total = 0
        for obj in self.world:
            sem_val = self.meaning(utterance, obj)
            item = tuple(obj.items())
            # probabilities[item] = math.exp(sem_val)
            probabilities[item] = math.exp(self.typicalityWeight*sem_val)
            total += math.exp(self.typicalityWeight*sem_val)

        # Normalize the semantic values
        for item in probabilities:
            probabilities[item] /= total
        return probabilities

    def cost(self, utt):
        return len(utt.split("_"))

    def pragmatic_speaker(self, obj, utterances):
        obj_key = tuple(obj.items())
        utterance_probs = {}
        total = 0.0
        for utt in utterances:
            literal_listener_prob = self.literal_listener(utt)
            utterance_prob = literal_listener_prob.get(obj_key)
            # Apply the pragmatic speaker function
            utility = self.alpha * math.log(utterance_prob) - self.costWeight * self.cost(utt)
            utterance_probs[utt] = math.exp(utility)
            total += math.exp(utility)

        # Normalize the values
        for item in utterance_probs:
            utterance_probs[item] /= total
        return utterance_probs


def singleton_overspecification_rate(
        beta_fixed=0.69, 
        state_semvalue_marked=0.95, 
        state_semvalue_unmarked=0.9, 
        costWeight=0
        ):
    overspecification_rates = []
    for condition in ["singleton_marked", "singleton_unmarked"]:
        this_world = parker_world[condition]
        utterances = [
            "door",
            "open_door",
            "closed_door",
            "other1",
            "other2",
            "other3"
        ]

        model = cs_rsa(
                this_world, 
                beta_fixed=beta_fixed, 
                state_semvalue_marked=state_semvalue_marked, 
                state_semvalue_unmarked=state_semvalue_unmarked, 
                costWeight=costWeight
                )
        results = model.pragmatic_speaker(this_world[0], utterances)
        print(results)
        overspecification_rate = sum(list(results.values())[1:3]) / sum(list(results.values())[:3])
        overspecification_rates.append(overspecification_rate)
    return overspecification_rates

def optimization(costWeight=0):
    target = np.array([0.24, 0.01])

    def objective(params):
        b, sm, su = params
        r_marked, r_unmarked = singleton_overspecification_rate(
            beta_fixed=float(b),
            state_semvalue_marked=float(sm),
            state_semvalue_unmarked=float(su),
            costWeight=costWeight
        )
        r = np.array([r_marked, r_unmarked], float)
        L = float(np.sum((r - target)**2))
        return L if np.isfinite(L) else 1e9

    bounds = [(0,1), (0,1), (0,1)]

    res_g = dual_annealing(objective, bounds=bounds, maxiter=1000)
    b, sm, su = res_g.x

    rates = singleton_overspecification_rate(
        beta_fixed=b,
        state_semvalue_marked=sm,
        state_semvalue_unmarked=su,
        costWeight=costWeight
    )

    print("=== Result ===")
    print(f"  beta_fixed           : {b:.4f}")
    print(f"  state_semvalue_marked: {sm:.4f}")
    print(f"  state_semvalue_unmarked: {su:.4f}")
    print(f"  resulting rates      : [{rates[0]:.4f}, {rates[1]:.4f}]")
    print(f"  target rates         : {target.tolist()}")
    print(f"  final loss           : {res_g.fun:.6e}")


    # # grid search
    # grid_pts = 50
    # step = 1.0 / (grid_pts - 1)
    # ranges = (slice(0.0, 1.0 + 0.5*step, step),   # include 1.0 robustly
    #           slice(0.0, 1.0 + 0.5*step, step),
    #           slice(0.0, 1.0 + 0.5*step, step))

    # x_brute, f_brute, _, _ = brute(objective, ranges,
    #                                full_output=True, finish=None)  # finish=None avoids NM
    # b2, sm2, su2 = np.clip(x_brute, 0, 1)
    # rates2 = singleton_overspecification_rate(beta_fixed=b2, state_semvalue_marked=sm2,
    #                                           state_semvalue_unmarked=su2, costWeight=costWeight)
    # print("\n=== Brute (grid) ===")
    # print(f"  beta_fixed              : {b2:.4f}")
    # print(f"  state_semvalue_marked   : {sm2:.4f}")
    # print(f"  state_semvalue_unmarked : {su2:.4f}")
    # print(f"  resulting rates         : [{rates2[0]:.4f}, {rates2[1]:.4f}]")
    # print(f"  target rates            : {target.tolist()}")
    # print(f"  final loss              : {f_brute:.6e}")

if __name__ == "__main__":
    print(singleton_overspecification_rate())

    # cs_rsa().meaning('open_door', {"size": "None", "state": "None", "nominal": "other1"}, print_value=True)

    # print(f"costWeight=0")
    # optimization(costWeight=0)
    # print(f"costWeight=2")
    # optimization(costWeight=2)

