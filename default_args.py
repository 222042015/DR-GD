def baseline_opt_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000

    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000

    defaults['qcqpVar'] = 100
    defaults['qcqpIneq'] = 50
    defaults['qcqpEq'] = 50
    defaults['qcqpEx'] = 10000

    if prob_type == ['simple', 'dcopf']:
        defaults['corrEps'] = 1e-4
    elif prob_type == 'nonconvex':
        defaults['corrEps'] = 1e-4
    elif prob_type == 'convex_qcqp':
        defaults['corrEps'] = 1e-4
    elif 'acopf' in prob_type:
        defaults['corrEps'] = 1e-4

    return defaults

def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 120
    
    # defaults['simpleVar'] = 2000
    # defaults['simpleIneq'] = 1000
    # defaults['simpleEq'] = 1000
    # defaults['simpleEx'] = 240
    
    # defaults['simpleVar'] = 1010
    # defaults['simpleIneq'] = 0
    # defaults['simpleEq'] = 11
    # defaults['simpleEx'] = 100
    
    # defaults['simpleVar'] = 3300
    # defaults['simpleIneq'] = 0
    # defaults['simpleEq'] = 301
    # defaults['simpleEx'] = 240
    defaults['earlyStop'] = 20

    if 'simple' in prob_type or 'port' in prob_type or 'emnist' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 3e-4                                                                                                                                                                                              
        defaults['hiddenSize'] = 128
        defaults['embSize'] = 128
        defaults['numLayers'] = 4
        defaults['lambda1'] = 1
        defaults['etaBase'] = 0.1
    elif 'control' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 5
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 1024
        defaults['embSize'] = 256
        defaults['numLayers'] = 4
        defaults['lambda1'] = 100
        defaults['etaBase'] = 0.1
    elif 'lasso' in prob_type:
        defaults['epochs'] = 100
        defaults['batchSize'] = 2
        defaults['lr'] = 1e-5
        defaults['hiddenSize'] = 128
        defaults['embSize'] = 128
        defaults['numLayers'] = 4
        defaults['lambda1'] = 1
        defaults['etaBase'] = 0.05
    elif 'qplib' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 1e-5
        defaults['hiddenSize'] = 128
        defaults['embSize'] = 128
        defaults['numLayers'] = 4
        defaults['lambda1'] = 1e2 #QPLIB_4270: 100000 # qplib_8906: 100
        defaults['etaBase'] = 0.05
    else:
        raise NotImplementedError

    return defaults


def l2ws_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 120
    defaults['earlyStop'] = 50

    if 'simple' in prob_type or 'port' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 1e-5
        # defaults['hiddenSize'] = [1000, 1000, 1000]
        defaults['hiddenSize'] = [500, 500, 500]

        defaults['embSize'] = None
        defaults['numLayers'] = None
        defaults['lambda1'] = None
        defaults['etaBase'] = None
        defaults['supervised'] = False
        defaults['train_unrolls'] = 0
    elif 'emnist' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 5
        defaults['lr'] = 1e-4
        # defaults['hiddenSize'] = [1000, 1000, 1000]
        defaults['hiddenSize'] = [784]

        defaults['embSize'] = None
        defaults['numLayers'] = None
        defaults['lambda1'] = None
        defaults['etaBase'] = None
        defaults['supervised'] = True
        defaults['train_unrolls'] = 0
    else:
        raise NotImplementedError

    return defaults