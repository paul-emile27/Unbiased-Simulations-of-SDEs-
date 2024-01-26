# The unbiased simulation algorithm
# The Path Dependent case

# This is basically a recursive implementation of the Markovian Case on each subintervals

import numpy as np

#np.random.seed(123)


# We introduce a random discrete time grid with β > 0 a fixed positive constant,
# (τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables on each subintervals.
def RandomTimeGrid_Interval(Beta, t1, t2):
    # Initialise the random time grid
    arr_t1t2 = [t1]
    sumTau = t1 + np.random.exponential(1/Beta)

    while sumTau < t2:
        arr_t1t2.append(sumTau)
        sumTau += np.random.exponential(1/Beta)

    # get Ntk_tilde := max{k : Tk < tk} - on the subintervals
    N_t1t2 = len(arr_t1t2)-1
    arr_t1t2.append(t2)

    return arr_t1t2, N_t1t2


# Simulate the Delta of the 1-dimensional Brownian motion W on a given subinterval
def BrownianMotionSimulation_Interval(Beta, t1, t2):
    # Get a random discrete time grid for the interval
    arr_t1t2, N_t1t2 = RandomTimeGrid_Interval(Beta, t1, t2)

    # Compute (DeltaT_k)k≥0
    arrDelta_t1t2 = np.diff(arr_t1t2)

    # Simulate the Delta of the 1d Brownian motion W
    arrDeltaW_t1t2 = np.zeros(N_t1t2 + 1)
    for i in range(N_t1t2 + 1):
        arrDeltaW_t1t2[i] = np.random.normal(loc=0.0, scale=np.sqrt(arrDelta_t1t2[i]))

    return N_t1t2, arr_t1t2, arrDelta_t1t2, arrDeltaW_t1t2


# The purpose of this function is to take care of the case
# μ(t,Xt1∧t,...,Xtn∧t), ie when the function μ differs in different subintervals
# However in the provided examples we did not make use of this function but feel free to adapt it in
# the Psi_US_1D_Recursive function changing line 82 by
#Xk_tilde[j+1] = Xk_tilde[j] + DeltaT_tkminus1_tk[j] * funcMu_k(k, Xk, arr_tkminus1_tk[j], Xk_tilde[j], len(lTimeIntervals), funcMu) + Sigma * DeltaW_tkminus1_tk[j]
# and line 90 by
#prodWk_tilde *= (funcMu_k(k, Xk, arr_tkminus1_tk[j], Xk_tilde[j], len(lTimeIntervals), funcMu) - funcMu_k(k, Xk, arr_tkminus1_tk[j-1], Xk_tilde[j-1], len(lTimeIntervals), funcMu)) * DeltaW_tkminus1_tk[j] / (DeltaT_tkminus1_tk[j] * Sigma)
def funcMu_k(k, Xk, t, Xj_tilde, numIter, funcMu):
    X = Xk.copy()
    for i in range(k, numIter):
        X.append(Xj_tilde)
    return funcMu(t, X)


#lTimeIntervals = (t1,..., tn)
def Psi_US_Recursive(k, Xk, X0, funcG, funcMu, Sigma, Beta, lTimeIntervals):

    # Sanity checks & Final condition of the recursive function
    if k == 0:
        raise ValueError("INPUT ERROR: k must start at 1")

    elif k == len(lTimeIntervals):
        return funcG(Xk[1:]) # May depend on the given funcG, here Xt1,...,Xtn

    # Get the random discrete time grid and the simulation of the Delta of the 1-dimensional Brownian motion W on each subintervals
    tk_minus1, tk = lTimeIntervals[k-1], lTimeIntervals[k]
    Nk_tilde, arr_tkminus1_tk, DeltaT_tkminus1_tk, DeltaW_tkminus1_tk = BrownianMotionSimulation_Interval(Beta, tk_minus1, tk)

    # Initialize array to store X_tilde values
    Xk_tilde = np.zeros(Nk_tilde + 2)

    # Set initial value of Xk_tilde from the simulated Xtk
    Xk_tilde[0] = Xk[-1]

    # local Euler scheme loop on [tk-1, tk]
    for j in range(Nk_tilde+1):
        # Euler scheme formula
        Xk_tilde[j+1] = Xk_tilde[j] + DeltaT_tkminus1_tk[j] * funcMu(arr_tkminus1_tk[j], Xk_tilde[j]) + Sigma * DeltaW_tkminus1_tk[j]

    if Nk_tilde > 0:
        # Initialize the products of the weights W^k_j of the estimator
        prodWk_tilde = 1
        # W^k_j loop
        for j in range(1, Nk_tilde + 1):
            # W^k_j formula
            prodWk_tilde *= (funcMu(arr_tkminus1_tk[j], Xk_tilde[j]) - funcMu(arr_tkminus1_tk[j-1], Xk_tilde[j-1])) * DeltaW_tkminus1_tk[j] / (DeltaT_tkminus1_tk[j] * Sigma)

        # Set the last values for the next recursive call
        Xk_0 = Xk.copy()
        Xk_0.append(Xk_tilde[-2])
        Xk.append(Xk_tilde[-1])

        return np.exp(Beta*(tk-tk_minus1))*(Psi_US_Recursive(k+1, Xk, X0, funcG, funcMu, Sigma, Beta, lTimeIntervals) - Psi_US_Recursive(k+1, Xk_0, X0, funcG, funcMu, Sigma, Beta, lTimeIntervals))*Beta**(-1*Nk_tilde)*prodWk_tilde

    # In the case N_T = 0
    Xk.append(Xk_tilde[-1])
    return np.exp(Beta*(tk-tk_minus1))*Psi_US_Recursive(k+1, Xk, X0, funcG, funcMu, Sigma, Beta, lTimeIntervals)


# We now provide a Monte Carlo estimation of the Unbiased Simulation Estimate
def MC_estimator(funcG, X0, funcMu, Sigma, Beta, lTimeIntervals, nSamples):

    psi_hats=np.zeros(nSamples)

    for i in range(nSamples):
        psi_hats[i] = Psi_US_Recursive(1, [X0], X0, funcG, funcMu, Sigma, Beta, lTimeIntervals)

    p=np.mean(psi_hats)
    s=np.std(psi_hats)

    #test, statistical confidence interval, statistical error
    return p, [p-1.96*s/np.sqrt(nSamples),p+1.96*s/np.sqrt(nSamples)], s/np.sqrt(nSamples)
