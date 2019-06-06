#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# ROCK, PAPER, SCISSORS respectively
ROCK, PAPER, SCISSORS = 0,1,2
NUM_ACTIONS = 3

def value(p1, p2):
    if p1==p2:
        return 0
    elif (p1-1) % NUM_ACTIONS == p2:
        return 1
    else:
        return -1

def normalise(strategy):
    strategy = strategy.copy()
    normalisingSum = np.sum(strategy)
    if normalisingSum > 0:
        strategy /= normalisingSum
    else:
        strategy = np.ones(NUM_ACTIONS)/NUM_ACTIONS
    return strategy

def getStrategy(regretSum):
    return normalise(np.maximum(regretSum, 0))

# Use regret-matching by randomly* selecting an action. *proportional to our positive regrets.
def getAction(strategy):
    strategy /= np.sum(strategy) #normalise
    return np.searchsorted(np.cumsum(strategy), random.random())

def innertrain(regretSum, strategySum, oppStrategy):
    # accumulate the current strategy based on regret
    strategy = getStrategy(regretSum)
    strategySum += strategy
    
    # regret-matching: choose action based on strategy
    myAction = getAction(strategy)
    oppAction = getAction(oppStrategy)
    
    actionUtility = np.zeros(NUM_ACTIONS)
    actionUtility[oppAction] = 0
    actionUtility[(oppAction + 1) % NUM_ACTIONS] = 1
    actionUtility[(oppAction - 1) % NUM_ACTIONS] = -1

    regretSum += actionUtility - actionUtility[myAction]
    
    return regretSum, strategySum

def train(iterations):
    regretSum = np.zeros(NUM_ACTIONS)
    strategySum = np.zeros(NUM_ACTIONS)
    oppStrategy = np.array([0.4,0.3,0.3])
    
    for i in range(iterations):
        regretSum, strategySum = innertrain(regretSum, strategySum, oppStrategy)
        
    return strategySum

def train2p(oiterations, iterations):
    strategySumP1 = np.zeros(NUM_ACTIONS)
    strategySumP2 = np.zeros(NUM_ACTIONS)
        
    for j in range(oiterations):
        regretSumP1 = np.zeros(NUM_ACTIONS)
        regretSumP2 = np.zeros(NUM_ACTIONS)
        
        oppStrategy = normalise(strategySumP2)
        for i in range(iterations):    
            regretSumP1, strategySumP1 = innertrain(regretSumP1, strategySumP1, oppStrategy)
            
        oppStrategy = normalise(strategySumP1)
        for i in range(iterations):
            regretSumP2, strategySumP2 = innertrain(regretSumP2, strategySumP2, oppStrategy)
        
        print(normalise(strategySumP1), normalise(strategySumP2))
        
    return strategySumP1, strategySumP2

s1, s2 = train2p(10, 1000)
normalise(s1), normalise(s2)

# 2 Player CFR Results
s1, s2 = train2p(100, 300)
strategy, oppStrategy = normalise(s1), normalise(s2)
vvv = []
for i in range(100):
    vv = 0
    for x in range(100):
        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)
        vv += value(myAction, otherAction)
    vvv.append(vv)
print(vvv[:10], vvv[90:])
plt.plot(sorted(vvv), 'r-', label='2 Player CFR'), np.mean(vvv), np.median(vvv)
#plt.hist(vvv, bins=5), np.mean(vvv)

# 1 Player CFR Results
stratSum = train(1000)
strategy = normalise(stratSum)
oppStrategy = np.array([0.4, 0.3, 0.3])

vvv = []
for i in range(100):
    vv = 0
    for x in range(100):
        myAction = getAction(strategy)
        otherAction = getAction(oppStrategy)
        vv += value(myAction, otherAction)
    vvv.append(vv)
print(vvv[:10], vvv[90:])
plt.plot(sorted(vvv), 'g-', label='1 Player CFR'), np.mean(vvv), np.median(vvv)
plt.legend()
plt.show()
#plt.hist(vvv, bins=5), np.mean(vvv)
