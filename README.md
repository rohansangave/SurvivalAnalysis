<h3>Nonlinear Semi-Parametric Models for Survival Analysis</h3>

<p>
  This repository contains code-base for https://arxiv.org/abs/1905.05865
</p>

<h5> Generating the results </h5>

<p> The results presented in the paper can be reproduced by executing the following command: <br>
  
  ```
  python MoCE.py
  python MoCE_results.py
  ```
 </p>

<h5>Generated Results </h5>

![ScreenShot](Images/Graphs.JPG)

<p align='center'> Figure 1: Concordance Index comparision on various dataset </p>

![ScreenShot](Images/LinearGating.JPG)
<p align='center'> Table 1:  Results of hard and soft linear gating networks and their comparison with relevantbaselines (95% bootstrap CI ). </p>


![ScreenShot](Images/NonLinearGating.JPG)
<p align='center'> Table 2:  Results of hard and soft non-linear gating networks and their comparison with relevantbaselines (95% bootstrap CI ). </p>

![ScreenShot](Images/HP_HardGating.JPG)
<p align='center'> Table 3: Hyperparameters for different datasets with hard-gating network</p>

![ScreenShot](Images/HP_Softgating.JPG)
<p align='center'> Table 4: Hyperparameters for different datasets with soft-gating network</p>


