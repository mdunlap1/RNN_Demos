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

## bptt.c
Implementation of epochwise back propagation through time for a single layer RNN.
Gradient calculation is verified by comparing to finite difference method. 

Compile with:
```make check_bptt```

Run with:
```./check_bptt```

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

## one_cell_lstm.c
Implementation of a single memory cell LSTM network and full (not truncated) epochwise back propagation through time. 

Compile with:
```make singe_cell```
Run with:
```./single_cell```

One can look at how perturbations to the weights (inputs) decay with time.
One can also look at how the error signal decays when propagating back (in time).
