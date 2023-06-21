## Week4

We initially tested our algorithm on a lunar lander and obtained satisfactory results. After approximately 10 minutes, we determined that the code should be suitable for Breakout.

![ll_anim](lunar_lander.gif)

However, we encountered difficulties in achieving satisfactory performance for Breakout. As a result, we referred to the DeepMind Atari implementation details available [here](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py), as well as a relevant example provided by Keras [here](https://keras.io/examples/rl/deep_q_network_breakout/) .

We observed that they employed reward shaping by assigning a negative reward for dying. Additionally, they fed grayscale images of the previous 4 frames to the model and utilized the Huber loss instead of the Mean Squared Error (MSE). After incorporating these modifications, our model began to perform better.

You can view the TensorBoard logs from our training run, which lasted approximately 8 hours, [here.](https://tensorboard.dev/experiment/NANVlF2nT8mLxiAQJ22Mrw/#scalars) Please note that the 'average_termination' metric indicates the average duration of an episode, and 'l2_diff' represents the L2 norm between the predictions of the Q network and the delayed Q network.

Below is an example episode from the later stages of our training:

![breakout_anim](breakout.gif)