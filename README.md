# Vaccination in Israel

This repository provides the Python scripts used for the analysis of vaccination data rom Israel. 

The Dynamic Survival Analysis (DSA) approach is based on this [white paper](https://idi.osu.edu/assets/pdfs/covid_response_white_paper.pdf), which is an extension of the methodology described in this [paper](https://royalsocietypublishing.org/doi/10.1098/rsfs.2019.0048). A Python implementation and detailed instructions on how to create the Python environment can be found [here](https://github.com/wasiur/dynamic_survival_analysis).

1. Run the following to get the parameter estimates from the September surge (Approach 2):
```bash
bash RegularDSA.sh
```
2. Run the following to get the parameter estimates for the first segment:
```bash 
bash ABC2_job_1.sh
```
3. Run the following to get the parameter estimates for the second segment:
```bash 
bash ABC2_job_2.sh
```
4. Run the following to get the parameter estimates for the third segment:
```bash 
bash ABC2_job_3.sh
```
5. Run the following to perform the counterfactual analyses and get the final figures:
```bash
python CombinedEffect.py
```

