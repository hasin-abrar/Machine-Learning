The task of channel equalization is to recover a sequence of transmitted bits at the receiver after they have
been distorted by the channel. From the theory of data communication, there are two sources of distortion
in the channel: (i) inter symbol interference where an information bit is affected by previously transmitted
bits, and (ii) noise, i.e., addition of unwanted signals in the channel.

In this project, the tasks done are:

1. Build a markov chain model for which one would need the following:
  * Defining the states.
  * Determining the state transition probabilities. Here one needs to find out all the possible transitions
among states and determine the respective probabilities.
  * Determining the observation probabilities. Since it is assumed the noise is normally distributed, the
observation probability should also follow normal distribution. Hence, we need only to find the mean of
the distributions

2. After the model is built, we need to test the model i.e., the test phase. Here one will transmit
a sequence of bits, and from the received l bits, xk, xk-1, ... xk-l+1 one needs to estimate the transmitted bit
sequence, Iˆk by using Viterbi algorithm on the Markov chain model.

3. Compare the original signal and the estimated signal to find out the accuracy.

4. To simulate the channel, one needs to define a class whose member will be the channel parameters: (i)
the h’s of the impulse response, (ii) mean and variance of channel noise. All these parameters are to be
taken as input from file. To simulate noise, one needs to generate a normal random variable with the given
mean and variance
