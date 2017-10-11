"""
routines to generate training and test sets
"""
import numpy as np
import parsers

def seperate_gips(gip,train_fraction):
    """
    split a single parsers.GeneralInputParser() into
    test and train parsers.GeneralInputParser()'s
    """
    num_structures = len(gip.supercells)

    train = parsers.GeneralInputParser()
    test = parsers.GeneralInputParser()
    train.supercells = []
    test.supercells = []

    # pick randomly
    train_idx = np.random.choice(np.arange(num_structures),size=round(train_fraction*num_structures),\
            replace=False)
    test_idx = set(np.arange(num_structures)).difference(set(train_idx))

    for ii,_s in enumerate(gip.supercells):
        if ii in train_idx:
            train.supercells.append(_s)
        elif ii in test_idx:
            test.supercells.append(_s)

    assert len(test.supercells)+len(train.supercells)==num_structures,'error in test/train splitting'           

    return {"train":train,"test":test}
