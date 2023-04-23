# Submition Week1

## Task 1

Not implemented


## Task 2


We decided to build a deterministic gridworld with walls and traps (which kill the agent). 

Our Policy $\pi(a|s)$ is defined like this:
1. The neighboring tile with the lowest Manhatten Distance is being chosen with a probability of 80%. We use argmin, so if there would be a tie, the first valid tile is being chosen.
2. The remaining 20% are evenly distributed over all the other valid neighbor tiles.

We use a penalzier per step of $0.1$ .

We interpretet one 'episode' like this:
1. Sample uniformly one of the tiles as starting tile  .
2. Calculate the reward of a trajectory starting starting from that tile. A trajectory stops if either a trap or the goal is bein reached.
3. Save the final performance of the trajectory for the starting tile to later calcute the mean for each tile. 


Here are the results after 50, 200, 500, 1000 & 10000 episodes each: 

<img src="imgs/initial_state_of_gridworld.jpg" alt="initial_state_of_gridworld"  width=50%>

<img src="imgs/after_50_iterations.jpg" alt="after_50_iterations"  width=50%>

<img src="imgs/after_200_iterations.jpg" alt="after_200_iterations"  width=50%>

<img src="imgs/after_500_iterations.jpg" alt="after_500_iterations"  width=50%>

<img src="imgs/after_1000_iterations.jpg" alt="after_1000_iterations"  width=50%>

<img src="imgs/after_10000_iterations.jpg" alt="after_10000_iterations"  width=50%>

An animation of one trajectory (a especially funny one):

![animation](ani.gif)
