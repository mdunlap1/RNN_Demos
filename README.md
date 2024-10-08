## Overview
Simple implementations of basic recurrent neural networks and original LSTM.
Intended for educational use only. 

## rtrl.c
Implementation of real time recrurrent learning for a single layer RNN. 
Gradient calculation is verified by comparing to finite difference method. 

Compile with: 
```make check_rtrl```

Run with:
```./check_rtrl```

Here is a graph of how network sensitivities change over time (subject to a low-pass filter):
![Sensitivities (low-pass filtered)](Images/Low_Pass_Filtered_Sensitivities.png)

## bptt.c
Implementation of epochwise back propagation through time for a single layer RNN.
Gradient calculation is verified by comparing to finite difference method. 

Compile with:
```make check_bptt```

Run with:
```./check_bptt```

Here is a graph of how the network activations are disturbed by perturbing the input. Here the input is Gaussian noise sampled from a standard normal distribution, and the perturbation is sufficient to saturate the activation.
Note how quickly the perturbed output "forgets" the perturbation and then matches the unperturbed output. 
![Perturbed Basic RNN](Images/Basic_RNN_Response_To_Perturbation.png)

## lstm.c
Implementation of an LSTM network as described in the original LSTM paper,
"Long Short-Term Memory" Hochreiter & Schmidhuber.
Specifically this is an implementation of the network for task 6a from that paper. 

Compile with:
```make task6a```

Run with:
```./task6a```

One can train the network on the task and see how many trials it takes to 
reach the stopping criteria. 
One an also train 30 such networks and get an everage of the number of trials it takes. 
One can also run the network in a REPL and see how each input affects the internal states of the network.
Before running in the REPL make sure to train a network and save the weights:
```./task6a -s```

Interestingly this code does not duplicate the results from the original paper. Perhaps there is a mistake in the implementation? Or perhaps there was a mistake in theirs? Or perhaps they had an unusually bad run? Here are some typical iteration counts to reach the stopping criteria:

```
30187
16885
18585
24194
21014
17430
19479
20492
27701
17727
20787
27946
23187
20921
17402
22578
18467
25792
27069
24715
```

## one_cell_lstm.c
Implementation of a single memory cell LSTM network and full (not truncated) epochwise back propagation through time. 

Compile with:
```make singe_cell```
Run with:
```./single_cell```

One can look at how perturbations to the weights (inputs) decay with time. Here is an example comparing perturbed and non-perturbed networks reponses to Gaussian (standard normal) input. 

![Perturbed Single Cell LSTM](Images/Single_Cell_LSTM_Response_To_Perturbation.png)

One can also look at how the error signal decays when propagating back (in time).
