import numpy as np
import warnings
from scipy.optimize import OptimizeResult
from deap import base,creator,tools,cma

def minimize(fun,jac,x0,args=(),solver='adam',tol=1e-8,**solver_kwargs):
    """
    Interface to stochastic global optimizers
    
    Examples
    --------
    >>> import nnp
    >>> 
    >>> adam_optimizer = nnp.optimizers.stochastic.minimize(fun=np.cos,jac=np.sin)
    """
    if solver.lower() not in ['adam','cma']:
        raise StochasticOptimizersError("{} is unsupported".format(solver))

    print('using optimizer {}'.format(solver))

    if solver.lower() == 'adam':
        optimizer = Adam(fun=fun,jac=jac,args=args,tol=tol,x0=x0,**solver_kwargs) 
    elif solver.lower() == 'cma':
        optimizer = CMA(fun=fun,args=args,x0=x0,**solver_kwargs)

    # run optimization
    optimisation_result = optimizer.minimize()

    # return final coordinates
    return optimisation_result

class Adam():
    """
    Adam stochastic optimizer

    Parameters
    ----------
    fun : callable function 
        Arguments are assumed to x
  
    jac : callable function
        Returns gradients of objective function wrt. x

    x0 : np.ndarray
        Initial coordinates

    tol : float
        Tolerance for algorithm termination

    Examples
    --------
    >>> import nnp
    >>> 
    >>> adam_optimizer = nnp.optimizers.stochastic.Adam(fun=np.cos,jac=np.sin)
    """
    
    def __init__(self,fun,jac,x0,args=(),learning_rate_init=1e-3,beta_1=0.9,beta_2=0.999,\
                 epsilon=1e-8,tol=1e-8,max_iter=100):
        self.fun = fun
        self.jac = jac
        self.x = x0
        self.args = args

        # stopping attributes
        self.tol = tol
        self.max_iter = max_iter

        # optimization info
        self.niter = 0
        self._no_improvement_count = 0
        self.loss_log = []
        self.best_loss = np.inf

        # algorithm parameters
        self.set_beta(beta_1,1)
        self.set_beta(beta_1,2)
        self.set_initial_learning_rate(learning_rate_init)
        self.epsilon = epsilon
        self.t = 0
        self.ms = np.zeros(self.x.shape[0])
        self.vs = np.zeros(self.x.shape[0]) 
        self.learning_rate = 0

    def set_beta(self,beta,identifier):
        """
        check beta value, must be [0,1)
        """
        if identifier not in [1,2]:
            raise StochasticOptimizersError("invalid identifier {}".format(identifier))

        if beta<0.0 or beta>=1.0:
            raise StochasticOptimizersError("invalid value of beta {}, must be [0,1)".\
                    format(beta))
        setattr(self,"beta_{}".format(int(identifier)),beta)
    
    def set_initial_learning_rate(self,learning_rate):
        self.learning_rate_init = learning_rate

    def _no_improvement_count(self):
        if self.loss_log[-1] > self.best_loss - self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if self.loss_log[-1] < self.best_loss:
            self.best_loss = self.loss_log[-1]
    
    def _get_updates(self):
        # compute jacobian
        jacobian = self.jac(self.x,self.args)

        self.t += 1
        self.ms = self.beta_1*self.ms + (1.0-self.beta_1)*jacobian
        self.vs = self.beta_2*self.vs + (1.0-self.beta_2)*(jacobian**2)
        self.learning_rate = self.learning_rate_init*np.sqrt(1.0-self.beta_2**self.t)\
                /(1.0-self.beta_1**self.t)
        update = -self.learning_rate*self.ms/(np.sqrt(self.vs)+self.epsilon)

        return update

    def update_params(self):
        """
        update current guess of optimal coordinates
        """
        self.x += self._get_updates()

    def minimize(self):
        """
        Perform minimization
        """
        self.niter = 0
        self.loss_log = []
        self.best_loss = np.inf
        self.t = 0

        batch_loss = np.zeros(1)

        opt_res = OptimizeResult()
        opt_res["success"] = False
        opt_res["nfev"] = 0
        opt_res["njev"] = 0

        for _it in range(self.max_iter):
            for ii,_batch in enumerate([1]):
                # iterate over mini batches of training data

                batch_loss[ii] = self.fun(self.x,self.args)
                opt_res["nfev"] += 1

                self.update_params()
                opt_res["njev"]
       
            # take average objective function over mini batches
            self.loss_log.append(np.average(batch_loss))
            opt_res["fun"] = self.loss_log[-1]

            # book keeping 
            self.niter += 1

            if self._no_improvement_count > 2:
                opt_res["success"] = True
                break
            if self.niter == self.max_iter:
                opt_res["status"] = "Stochastic optimization reached max. iterations"
                warnings.warn('Stochastic optimization reached max iterations {} and did not converge'.\
                        format(self.max_iter))
        opt_res["x"] = self.x
        opt_res["niter"] = self.niter

        return opt_res
        
class CMA():
    """
    Covariance Matrix Adaption

    Parameters
    ----------
    fun : Callable function
        Objective function to minimise
    """

    def __init__(self,fun,x0,args=(),sigma=1e-3,max_iter=1000):        
        self.fun = fun
        self.args = args
        self.x0 = x0
        # generate wrapped function
        self._set_function()

        self.sigma = sigma
        self.max_iter = max_iter

        # assume length of input corresponds to num params 
        if isinstance(x0,(np.ndarray,tuple,list)):
            self.Nparam = np.shape(x0)[0]
        else:
            self.Nparam = 1

    def _set_function(self):
        def wrapped_function(x):
            return (self.fun(x,self.args),)

        self._function = wrapped_function

    def minimize(self):

        sigma = self.sigma
        lambda_ = int(4+3*np.log(self.Nparam))
        
        creator.create("FitnessMin",base.Fitness,weights=(-1.,))
        creator.create("Individual",list,fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("map",map)
        toolbox.register("evaluate",self._function)

        strategy = cma.Strategy(centroid=self.x0,sigma=sigma)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update",strategy.update)


        #-----------------#
        # log information #
        #-----------------#
        
        halloffame = tools.HallOfFame(maxsize=1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg",np.mean)
        stats.register("std",np.std)
        stats.register("min",np.min)
        stats.register("max",np.max)

        logbook = tools.Logbook()
        logbook.header = "gen","evals","avg","std","min","max"

        conditions = {"MaxIter" : False, "TolHistFun" : False, "EqualFunVals" : False,
                      "TolX" : False, "TolUpSigma" : False, "Stagnation" : False,
                      "ConditionCov" : False, "NoEffectAxis" : False, "NoEffectCoor" : False,
                       "small_std":False}

        MAXITER = self.max_iter
        
        # return a ScipyOptimize object
        opt_res = OptimizeResult()
        opt_res["success"] = False
        opt_res["nfev"] = 0

        t = 0
        while not any(conditions.values()):
            # generate all indidivuals
            population = toolbox.generate()

            # evaluate fitnesses for all individuals in population
            fitnesses = toolbox.map(toolbox.evaluate,population)

            for ind,fit in zip(population,fitnesses):
                ind.fitness.values = fit

            # update records of optimisation
            halloffame.update(population)
            record = stats.compile(population)
            logbook.record(gen=t,evals=lambda_,**record)

            # number of function evaluations
            opt_res["nfev"] += len(population)

            toolbox.update(population)

            # book keeping 
            t += 1

            if t > MAXITER:
                conditions["MaxIter"] = True

        # "best" individual of all time
        opt_res["fun"] = np.min([_pop["min"] for _pop in logbook])
        opt_res["x"] = halloffame[0]
        opt_res["logbook"] = logbook

        # return Scipy OptimizeResult instance 
        return opt_res
        
        
class StochasticOptimizersError(Exception):
    pass    
