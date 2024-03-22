# Hyperparameter selection for physics-informed neural networks (PINNs) – Application to discontinuous heat conduction problems
In recent years, physics-informed neural networks (PINNs) have emerged as an alternative to conventional numerical techniques to solve forward and inverse problems involving partial differential equations (PDEs). Despite its success in problems with smooth solutions, implementing PINNs for problems with discontinuous boundary conditions (BCs) or discontinuous PDE coefficients is a challenge. The accuracy of the predicted solution is contingent upon the selection of appropriate hyperparameters. In this work, we performed hyperparameter optimization of PINNs to find the optimal neural network architecture, number of hidden layers, learning rate, and activation function for heat conduction problems with a discontinuous solution. Our aim was to obtain all the settings that achieve a relative L2 error of 10% or less across all the test cases. Results from five different heat conduction problems show that the optimized hyperparameters produce a mean relative L2 error of 5.60%.

## Libraries
* NVIDIA Modulus 22.03 (not 22.03.1) from NVIDIA NGC
* Hydra
* PyTorch

## Citation
```
@article{sharma2023hyperparameter,
  title={Hyperparameter selection for physics-informed neural networks (PINNs)--Application to discontinuous heat conduction problems},
  author={Sharma, Prakhar and Evans, Llion and Tindall, Michelle and Nithiarasu, Perumal},
  journal={Numerical Heat Transfer, Part B: Fundamentals},
  pages={1--15},
  year={2023},
  publisher={Taylor \& Francis}
}
```
