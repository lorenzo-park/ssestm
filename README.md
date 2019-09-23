How to use with Anaconda environment (Recommended)

1. conda create -n ssestm python=3.7

2. conda activate ssestm

3. pip install -r requirements.txt

3. python ssestm.py run --alpha_plus 0.2 --alpha_minus 0.2 --kappa 3
                 --reg 0.001 --alpha_rate 0.0001 --max_iters 1000 --error 0.000001 --skip_params_gen False
                 
4. If you have done 3. before, then in the next time, you can run it by just 
python ssestm.py run