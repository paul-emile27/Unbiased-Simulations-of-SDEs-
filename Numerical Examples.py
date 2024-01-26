# In this section we call the different algorithms to provide our numerical tests
# and computation times

import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

import Euler_Scheme
import Markovian_Case
import Path_Dependent_Case


# function to convert time into hours, minutes, and seconds
def convert_to_hms(seconds):
    # Extract whole seconds and fractional part
    whole_seconds = int(seconds)
    fractional_seconds = seconds - whole_seconds

    # Convert whole seconds to a timedelta object
    time_delta = datetime.timedelta(seconds=whole_seconds)

    # Extract hours, minutes, and remaining whole seconds
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)

    # Combine whole seconds with fractional part and round to 6 decimals
    precise_seconds = round(whole_seconds + fractional_seconds, 6)

    # Format the result
    result = ""
    if hours > 0:
        result += f"{hours}h "
    if minutes > 0 or hours > 0:  # Include minutes if there are hours
        result += f"{minutes}min "
    result += f"{precise_seconds}s"

    return result.strip()



##### TEST for V0 in (4.2) (expected result : 0.205396 around) #####


# Parameters

X0 = 0  # Initial value
T = 1   # Maturity
nDim = 1    # Dim of process
mSteps = 10     # Number of time steps in Euler Scheme
nSamples = 10**5   # Number of simulations of MC

K = 1   # Strike
Sigma0 = 0.5
Beta = 0.1  # Beta constant
M = 4   # M constant

lTimeIntervals = [0, T]

# μ in the provided SDE
def funcMu(t, x):
    return 0.1 * (np.sqrt(np.exp(x)) - 1) - 0.125
# Payoff G in the provided example (Call option)
def funcG(x):
    return np.maximum(0, np.exp(x) - K)
# Payoff G in the provided example (Call option) for the Path Dependent Case
def funcG_PathDep(x):
    return np.maximum(0, np.exp(x[-1]) - K)



# Run The Simulations

print("RESULTS FOR THE MARKOVIAN EXAMPLE 4.2 (expected result : 0.205396 around)")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, Sigma0, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Markovian:", estimator)
print("95% Confidence Interval MC_EulerScheme_Markovian:", confidence_interval)
print("Standard Error MC_EulerScheme_Markovian:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, Beta, T, nDim, nSamples)
print("Estimator US_Markovian_Case:", estimator)
print("95% Confidence Interval US_Markovian_Case:", confidence_interval)
print("Standard Error US_Markovian_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")



Method = []
Mean_value = []
conf_interval = []
statistical_error = []
Computation_time = []
Estimated_bias = []
for i in range(4, 9):
    nSamples = 10 ** i

    start_time = time.time()
    estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, Beta, T, nDim, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"US (N = 10^{i})")

    start_time = time.time()
    estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, Sigma0, T, nDim, mSteps, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"Euler Scheme (nSteps = 10^{i})")

for i in range(len(Mean_value)):
    Estimated_bias.append(Mean_value[i] - Mean_value[-2])


# Round numbers and make sure confidence it fits in the cell
rounded_mean_value = [round(val, 8) for val in Mean_value]
rounded_statistical_error = [round(val, 9) for val in statistical_error]
formatted_computation_time = [convert_to_hms(val) for val in Computation_time]
formatted_conf_interval = [f"[{round(ci[0], 8)}, {round(ci[1], 8)}]" for ci in conf_interval]
rounded_Estimated_bias = [round(val, 9) for val in Estimated_bias]

# Sample data:
data = {
    'Method': Method,
    'Mean value': rounded_mean_value,
    'Statistical error': rounded_statistical_error,
    '95% Confidence Interval': formatted_conf_interval,
    'Computation time': formatted_computation_time
}

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Adjust the figure size (width, height) to accommodate the data
fig, ax = plt.subplots(figsize=(20, 8))  # You may need to adjust these values

# Hide the axes
ax.axis('off')

# Determine column widths - increase width for confidence interval
colWidths = [0.15, 0.1, 0.1, 0.2, 0.1]

# Create the table with adjusted settings
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["palegreen"] * len(df.columns),
                     colWidths=colWidths)

# Adjust font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust the size as needed

# Scale the table to the figure by setting its dimensions
the_table.scale(1, 1.5)  # The second value increases the row heights

# Tight layout for a neat fit
plt.tight_layout()

# Save the table as an image file
plt.savefig('C:/Users/natha/OneDrive/Bureau/MASEF/S1/MC methods FE applied fi/Numerical results/Markovian Numerical results plot.png')

# Sample data
data = {
    'Method': Method,
    'Mean value': rounded_mean_value,
    'Estimated Bias': rounded_Estimated_bias,
    'Statistical error': rounded_statistical_error,
}

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Adjust the figure size (width, height) to accommodate the data
fig, ax = plt.subplots(figsize=(20, 8))  # You may need to adjust these values

# Hide the axes
ax.axis('off')

# Determine column widths - increase width for confidence interval
colWidths = [0.15, 0.1, 0.1, 0.2, 0.1]

# Create the table with adjusted settings
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["palegreen"] * len(df.columns),
                     colWidths=colWidths)

# Adjust font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust the size as needed

# Scale the table to the figure by setting its dimensions
the_table.scale(1, 1.5)  # The second value increases the row heights

# Tight layout for a neat fit
plt.tight_layout()

# Save the table as an image file
plt.savefig('C:/Users/natha/OneDrive/Bureau/MASEF/S1/MC methods FE applied fi/Numerical results/Markovian Bias Numerical results plot.png')


##### TEST for V0_tilde in (4.2) (expected result : 0.1267 around) #####


# Parameters:

Beta = 0.05 # Beta constant
lTimeIntervals = [i*T/10 for i in range(0, 11)]

# We adapt the new path dependent payoff to the example
def funcG_PathDep (lX):
    return np.maximum(0, np.sum(np.exp(lX))/len(lX) - K)



# Run The Simulations

print("RESULTS FOR THE PATH DEPENDENT EXAMPLE 4.2 (expected result : 0.1267 around)")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Pathdep(funcG_PathDep, X0, funcMu, Sigma0, T, nDim, mSteps, nSamples)
print("Estimator MC_estimator_EulerScheme_Pathdep:", estimator)
print("95% Confidence Interval MC_EulerScheme_Pathdep_Example::", confidence_interval)
print("Standard Error MC_EulerScheme_Pathdep_Example::", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")

start_time = time.time()
estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
print("Estimator US_Path_Dependent_Case:", estimator)
print("95% Confidence Interval US_Path_Dependent_Case:", confidence_interval)
print("Standard Error US_Path_Dependent_Case:", error)
print(f"Execution time: {time.time() - start_time} seconds")
print(" ")




Method = []
Mean_value = []
#conf_interval = []
statistical_error = []
Computation_time = []
Estimated_bias = []
for i in range(4, 8):
    nSamples = 10 ** i

    start_time = time.time()
    estimator, confidence_interval, error = Path_Dependent_Case.MC_estimator(funcG_PathDep, X0, funcMu, Sigma0, Beta, lTimeIntervals, nSamples)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    #conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"US (N = 10^{i})")

    start_time = time.time()
    estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Pathdep(funcG_PathDep, X0, funcMu, Sigma0, T, mSteps, nSamples, lTimeIntervals)
    Computation_time.append(time.time() - start_time)
    Mean_value.append(estimator)
    #conf_interval.append(confidence_interval)
    statistical_error.append(error)
    Method.append(f"Euler Scheme (N = 10^{i})")

for i in range(len(Mean_value)):
    Estimated_bias.append(Mean_value[i] - Mean_value[-2])

# Round numbers and make sure confidence it fits in the cell
rounded_mean_value = [round(val, 8) for val in Mean_value]
rounded_statistical_error = [round(val, 9) for val in statistical_error]
formatted_computation_time = [convert_to_hms(val) for val in Computation_time]
rounded_Estimated_bias = [round(val, 9) for val in Estimated_bias]
#formatted_conf_interval = [f"[{round(ci[0], 8)}, {round(ci[1], 8)}]" for ci in conf_interval]

# Sample data:
data = {
    'Method': Method,
    'Mean value': rounded_mean_value,
    'Statistical error': rounded_statistical_error,
    'Estimated Bias': rounded_Estimated_bias,
    'Computation time': formatted_computation_time
}

# Convert data to a Pandas DataFrame
df = pd.DataFrame(data)

# Adjust the figure size (width, height) to accommodate the data
fig, ax = plt.subplots(figsize=(20, 8))  # You may need to adjust these values

# Hide the axes
ax.axis('off')

# Determine column widths - increase width for confidence interval
colWidths = [0.15, 0.1, 0.1, 0.2, 0.1]

# Create the table with adjusted settings
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["palegreen"] * len(df.columns),
                     colWidths=colWidths)

# Adjust font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)  # Adjust the size as needed

# Scale the table to the figure by setting its dimensions
the_table.scale(1, 1.5)  # The second value increases the row heights

# Tight layout for a neat fit
plt.tight_layout()

# Save the table as an image file
plt.savefig('C:/Users/natha/OneDrive/Bureau/MASEF/S1/MC methods FE applied fi/Numerical results/Path Dependent Numerical results plot.png')


##### TEST for V0 in (4.3) - Building the graph of Computation time with Beta #####
# Parameters

X0 = 0  # Initial value
T = 1   # Maturity
nDim = 1    # Dim of process
mSteps = 10 # Number of time steps in Euler Scheme
nSamples = 10**5   # Number of simulations of MC

Sigma0 = 0.5

# Payoff G in the provided example sin(X_T)
def funcG(x):
    return np.sin(x)
# μ in the provided SDE
def funcMu(t, x):
    return 0.2 * np.cos(x)


#beta_values = [0.025*i for i in range(1, 250, 2)] #0.025
#beta_values = [0.05*i for i in range(1, 10)]
beta_values = [0.005*i for i in range(1, 101)]

US_CompTime = []
EulerScheme_CompTime = []
US_Var = []
EulerScheme_Var = []

start_time = time.time()
estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, Sigma0, T, nDim, mSteps, nSamples)
EulerScheme_t = time.time() - start_time
EulerScheme_Var.append((error*np.sqrt(nSamples))**2)

for beta in beta_values:
    start_time = time.time()
    estimator, confidence_interval, error = Markovian_Case.MC_estimator(funcG, X0, funcMu, Sigma0, beta, T, nDim, nSamples)
    US_CompTime.append(time.time() - start_time)
    US_Var.append((error*np.sqrt(nSamples))**2)
    EulerScheme_CompTime.append(EulerScheme_t)

    estimator, confidence_interval, error = Euler_Scheme.MC_estimator_EulerScheme_Markovian(funcG, X0, funcMu, Sigma0, T, nDim, mSteps, nSamples)
    EulerScheme_Var.append((error*np.sqrt(nSamples))**2)

plt.clf()
# Plot
fig, ax1 = plt.subplots()

# Plot the US Computation Time on the primary y-axis
ax1.plot(beta_values, US_CompTime, 'g-', label='US Computation Time')
ax1.set_xlabel('Beta')
ax1.set_ylabel('Computation time in seconds', color='g')
ax1.tick_params('y', colors='g')

# Plot the Euler Scheme Computation Time on the primary y-axis as well
ax1.plot(beta_values, EulerScheme_CompTime, 'r--', label='Euler Scheme Computation Time')

# Add a legend
ax1.legend(loc='upper left')

# Title
plt.title('Evolution of Computation Time of US given Beta')

# Save the figure
plt.savefig('Write/Your/Path/Here', dpi=300, bbox_inches='tight')


plt.clf()
fig, ax1 = plt.subplots()

# Plotting the US Computation Time
ax1.plot(beta_values, US_CompTime, 'g-', label='US Computation Time')
ax1.set_xlabel('Beta')
ax1.set_ylabel('Computation time in seconds', color='g')
ax1.tick_params('y', colors='g')

# Creating a secondary axis for Variance
ax2 = ax1.twinx()
ax2.plot(beta_values, US_Var, 'b-', label='US Variance')
ax2.plot(beta_values, EulerScheme_Var, 'r-', label='Euler Scheme Variance')  # Updated label
ax2.set_ylabel('Variance', color='b')
ax2.tick_params('y', colors='b')

# Updating the legend to include all plots
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')


# Title
plt.title('Evolution of the Computation Time and the Variance of US given small Beta')

# Save the figure
plt.savefig('Write/Your/Path/Here', dpi=300, bbox_inches='tight')
