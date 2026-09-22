# Running Scripts

Since we are on a login node, we have to

1) Start a screen session
2) Check the jobid using squeue
3) srun --jobid=<> --overlap --pty bash to use the gpu
4) Run whatever script I told you about
5) Once the screen is done running the script, kill the screen