# The unbiased simulation algorithm
# The Markovian case for General SDEs

import numpy as np


#np.random.seed(123)

def funcA(t, x, funcSigma):
    return(0.5 * np.dot(funcSigma(t, x), funcSigma(t, x).T))


# We introduce a random discrete time grid with β > 0 a fixed positive constant,
# (τ_i)i>0 be a sequence of i.i.d. E(β)-exponential random variables.
def RandomTimeGrid(Beta, T):
    # Initialise the random time grid
    lT = [0]
    sumTau = np.random.exponential(1/Beta)

    while sumTau < T:
        lT.append(sumTau)
        sumTau += np.random.exponential(1/Beta)

    # get Nt := max{k : Tk < t}
    N_T = len(lT)-1
    lT.append(T)

    return lT, N_T



def Unbiased_Simulation_Markovian_Case_GeneralSDEs(funcG, X0, funcMu, funcSigma, Beta, nDim, T):
    # Get a random discrete time grid
    arrTimeGrid, N_T = RandomTimeGrid(Beta, T)

    # Compute (DeltaT_k)k≥0
    arrDeltaT = np.diff(arrTimeGrid, axis = 0)

    # Initialize array to store X_hat values
    X_hat = np.zeros((N_T + 2, nDim))

    # Set initial value
    X_hat[0] = X0

    # Simulate the Delta of the d-dimensional Brownian motion W
    arrDeltaW = np.zeros((N_T+1, nDim))
    for i in range(N_T + 1):
        arrDeltaW[i] = np.random.normal(loc=0.0, scale=np.sqrt(arrDeltaT[i]), size=nDim)

    # Euler scheme loop
    for k in range(N_T+1):
        # Euler scheme formula
        X_hat[k + 1] = X_hat[k] + arrDeltaT[k] * funcMu(arrTimeGrid[k], X_hat[k]) + np.dot(Sigma, arrDeltaW[k])

    if N_T > 0:
        # Initialize the products of the automatic weights
        prodW1_W2 = 1

        # W^1_k + W^2_k loop
        for k in range(1, N_T+1):
            # W^1_k formula
            W1 = ((funcMu(arrTimeGrid[k], X_hat[k]) - funcMu(arrTimeGrid[k-1], X_hat[k-1]))*arrDeltaW[k])/(arrDeltaT[k]*funcSigma(arrTimeGrid[k], X_hat[k]))

            # W^2_k formula
            A = funcA(arrTimeGrid[k], X_hat[k], funcSigma) - funcA(arrTimeGrid[k-1], X_hat[k-1], funcSigma)
            B = np.linalg.inv(funcSigma(arrTimeGrid[k], X_hat[k]).T)*(np.dot(arrDeltaW[k], arrDeltaW[k].T)-arrDeltaT[k]*np.eye(nDim))/(arrDeltaT[k]*arrDeltaT[k]) * np.linalg.inv(funcSigma(arrTimeGrid[k], X_hat[k]))
            W2 = np.trace(np.dot(A, B.T))

            prodW1_W2 *= W1 + W2

        # Compute the estimator
        return np.exp(Beta*T)*(funcG(X_hat[-1]) - funcG(X_hat[N_T]))*Beta**(-1*N_T)*prodW1_W2

    # Compute the estimator in the case N_T = 0
    return np.exp(Beta*T)*funcG(X_hat[-1])*Beta**(-1*N_T)



# We now provide a Monte Carlo estimation of the Unbiased Simulation Estimate
def MC_estimator(funcG, X0, funcMu, funcSigma, Beta, T, nDim, nSamples):

    psi_hats = np.zeros(nSamples)
    
    for i in range(nSamples):
        psi_hats[i] = Unbiased_Simulation_Markovian_Case_GeneralSDEs(funcG, X0, funcMu, funcSigma, Beta, nDim, T)

    p = np.mean(psi_hats)
    s = np.std(psi_hats)

    #test, statistical confidence interval, statistical error
    return p, [p-1.96*s/np.sqrt(nSamples),p+1.96*s/np.sqrt(nSamples)], s/np.sqrt(nSamples)
