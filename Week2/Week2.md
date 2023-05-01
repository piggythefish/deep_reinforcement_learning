# Submition Week2

## Task 2


We decided changed the our deterministic gridworld to an probabilistic. 

We use a deterministic Policy $\pi(a|s)$, which we initialize randomly at the start.

Our probabilistic state dynamics $p(R_{t+1}|s,a)$ are controlled by the new argument ```probality_wrong_step```, ```which dictates the probability of taking the agent misstepping into a different direction then intended. If an agent missteps, the new direction will be uniformally sampled.  


For the (probabilistic) reward dynamics $p(R_{t+1}|s,a)$ we calculate the average returns of each state action pair.

Our Hyperparameters were:

- ```trap_reward =-1,```
- ```goal_reward = 3,```
- ```cost_per_step = 0.05,```
- ```max_steps_per_episode =200```
- $\gamma$ = 0.9

This is how an initial state of the bord looks like:



<img src="imgs/initial_state_of_gridworld.jpg" alt="initial_state_of_gridworld"  width=50%>


We did 10000 iterations of MC Policy Iteration for two differetn probality_wrong_steps [Failure or Missstep rates]. We used 5% and 10%, which resulted in quite interesting differences. 


### The Policies After 10000 Steps 

<img src="imgs/after_10000_steps_-_5_failure_rate.jpg" alt="sadsa"  width=50%>


<img src="imgs/after_10000_steps_-_10_failure_rate.jpg" alt="sadsa"  width=50%>

What is interesting here is that the 5% Failure Rate leads to generally higher rewards further away and less good policys on the less visited corners.

On the other hand the 10% Failure Rate has generally lower rewards but better policie decisions at the corners.

###  The average Return-per-Episodes

<img src="imgs/0.05_failure_rate_over_time.jpg" alt="after_200_iterations"  width=50%>

<img src="imgs/0.1_failure_rate_over_time.jpg" alt="after_200_iterations"  width=50%>

Here again see that a lower Failure Rate leeds to higher returns.

### Number of samples for each Tile

<img src="imgs/sample_counts_5.jpg" alt="after_500_iterations"  width=50%>
<img src="imgs/sample_counts_10.jpg" alt="after_500_iterations"  width=50%>

These images show how often each tile was visited. What we see is that for the 5% Failure Rate even after 10000 episodes some tiles were only visited 20 times, which explains the bad decision there.

The 10% Failure Rate leeds to far of tiles being visited more and thus better policies at the corners.

###  One Episode with 10% Failure Rate after 10000 Iterations

![episode](videos/episode.gif)

### State-Action Values over Time with 10% Failure Rate for different Steprates

![one](videos/7000_Steps.gif)


![ji](videos/20000_Steps.gif)
