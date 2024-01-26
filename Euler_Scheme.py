# In this section we define the different Euler discretization Schemes
# we are using for the different examples

import numpy as np

# Euler scheme for simulating SDEs over a fixed time grid.
def Euler_Scheme(X0, funcMu, Sigma0, T, mSteps):
    #time step size
    dt = T / mSteps
    X = np.zeros(mSteps+1)

    # the Euler scheme at X0
    X[0] = X0
    # Get the grid (t0,...,tm=T) with steps dt
    time_grid = np.linspace(0, T, mSteps + 1)

    # Euler scheme loop
    for i in range(mSteps):
        # Euler scheme formula
        X[i+1] = X[i] + funcMu(time_grid[i], X[i])*dt + Sigma0 * np.random.normal(loc=0.0, scale=np.sqrt(dt))

    return X


# We now provide a Monte Carlo estimation of Euler Scheme in the Markovian Case
def MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, Sigma0, T,nDim,mSteps, nSamples):

    g_hats = np.zeros(nSamples)

    for i in range(nSamples):
        g_hats[i] = funcG(Euler_Scheme(X0, funcMu, Sigma0, T, mSteps)[-1]) #Get the last element of the Euler-Scheme

    p = np.mean(g_hats)
    s = np.std(g_hats)

    #mean-value, statistical confidence interval, statistical error
    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s/np.sqrt(nSamples)], s / np.sqrt(nSamples)


# We now provide a Monte Carlo estimation of Euler Scheme for a path-dependent payoff
def MC_estimator_EulerScheme_Pathdep(funcG, X0, funcMu, Sigma0, T,mSteps, nSamples, lTimeIntervals):

    g_hats = np.zeros(nSamples)
    step_size = mSteps // (len(lTimeIntervals)-1) #get the right step for getting (t1,...,tn)

    for i in range(nSamples):
        g_hats[i] = funcG(Euler_Scheme(X0, funcMu, Sigma0, T, mSteps)[step_size::step_size])

    p = np.mean(g_hats)
    s = np.std(g_hats)

    #mean-value, statistical confidence interval, statistical error
    return p, [p - 1.96 * s / np.sqrt(nSamples), p + 1.96 * s / np.sqrt(nSamples)], s / np.sqrt(nSamples)
