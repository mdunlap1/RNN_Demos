objects = matrix functions

check_rtrl : rtrl.c $(objects)
	gcc -lm -lgsl $(objects) -o check_rtrl rtrl.c

check_bptt : bptt.c $(objects)
	gcc -lm -lgsl $(objects) -o check_bptt bptt.c

matrix : matrix.c
	gcc -c -o matrix matrix.c

functions : functions.c 
	gcc -c -lm -lgsl -o functions functions.c

task6a : lstm.c $(objects)
	gcc -lm -lgsl $(objects) -o task6a lstm.c

single_cell : one_cell_lstm.c $(objects)
	gcc -lm -lgsl -g $(objects) -o single_cell one_cell_lstm.c


clean:
	rm -f check_rtrl check_bptt task6a single_cell $(objects)
