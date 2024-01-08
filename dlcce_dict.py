import numpy as np
import pickle

# train energy
e_train_dict = {
    "alexnet": 23.82893,
    "squeezenet": 55.5445,
    "resnet50": 101.33996,
    "vgg16": 64.033884
}

# inference energy
e_inf_dict = {
    "agx64g": {
        "alexnet": 1.86,
        "squeezenet": 3.95,
        "resnet50": 13.24,
        "vgg16": 16.46
    },
    "nano8g": {
        "alexnet": 3.84,
        "squeezenet": 3.08,
        "resnet50": 12.99,
        "vgg16": 16.46
    },
    "cpu64g": {
        "alexnet": 59.05,
        "squeezenet": 197.51,
        "resnet50": 325.46,
        "vgg16": 414.20
    }
}

# F ranges from 100 to 1000000000
F_values = np.linspace(100, 1000000000, 10000)

def dlcce(train, test, F_values):
    dlcce_dict = {key: {sub_key: {'dlcce': [], 'e_inf': 0} for sub_key in e_inf_dict[key].keys()} for key in e_inf_dict}
    for dev, val in test.items():
        for mod, inf in val.items():
            train_e = train[mod]
            dlcce = 1 / (train_e*(10**6) / (inf * F_values) + 1)
            dlcce_dict[dev][mod]["dlcce"].extend(dlcce)
            dlcce_dict[dev][mod]["e_inf"] = inf
    return dlcce_dict

dlcce_ = dlcce(e_train_dict, e_inf_dict, F_values)

with open('dlcce.pkl', 'wb') as f:
    pickle.dump(dlcce_, f)