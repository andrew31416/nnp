import nnp.features.types


def Behler(train_data):
    defaults = nnp.features.types.features(train_data=train_data,PCA=None)

    rcut = 6.5
    rs = 1.0

    for _eta in [0.05,4.0,20.0,80.0]:
        defaults.add_feature(nnp.features.types.feature("acsf_behler-g2",{"rcut":rcut,\
                "fs":0.1,"eta":_eta,"rs":rs,"za":1.0,"zb":1.0}))

    for _lambda in [-1.0,1.0]:
        for _xi in [1.0,4.0]:
            defaults.add_feature(nnp.features.types.feature("acsf_behler-g4",\
                    {"rcut":rcut,"fs":0.1,"eta":0.005,"xi":_xi,"lambda":_lambda,\
                    "za":1.0,"zb":1.0}))

    return defaults            
