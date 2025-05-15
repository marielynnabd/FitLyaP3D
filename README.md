# FitLyaP3D

FitLyaP3D is a Python pipeline for performing **cosmological interpretation** of Lyman-alpha forest **Px** and **P3D** measurements. It builds on the output of the [LyaP3D](https://github.com/marielynnabd/LyaP3D) pipeline by fitting measurements to theoretical models using statistical inference methods.

## Features

- Fits Px and P3D measurements to theoretical models or emulator predictions
- Supports two modeling options:
  - **Analytical model** based on Arinyo-i-Prats et al. (2015) (https://arxiv.org/abs/1506.04519)
  - **Emulator-based model** using the ForestFlow Emulator (https://github.com/igmhub/ForestFlow) (https://arxiv.org/abs/2409.05682)
- Performs parameter estimation using **$\chi^2$ minimization** with Iminuit (https://iminuit.readthedocs.io/)
- Outputs best-fit cosmological and astrophysical parameters with uncertainties
- Example Bash and Python execution scripts available upon request
