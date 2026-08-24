# Cox Processes with Exogenous AI Modulation

## Identifiability, Stability, and Approximation in Law

This repository contains the computational material supporting the manuscript

> **Cox Processes with Exogenous AI Modulation: Identifiability, Stability, and Approximation in Law**

by **Dr. Debashis Chatterjee**  
Department of Statistics, Visva-Bharati University, Santiniketan, West Bengal, India.

The paper develops a probability-theoretic framework for **AI-modulated doubly stochastic Poisson processes (AIM-DSPPs)**. The central construction is


$\lambda_\Theta(t)=R_tA_\Theta(t),$


where \(R_t\) is a nonnegative random Cox directing intensity and \(A_\Theta(t)\) is a positive learned modulation field driven by **exogenous** information. Conditional on the directing sigma-field, the event process remains Poisson.

The repository is intended to make the simulation study and the two real-data analyses reproducible.

---

## Main theoretical themes

The manuscript studies the following structural questions.

- **Cox-preserving learned modulation.** The learned component is introduced only through exogenous directing information, preserving conditional-Poisson structure.
- **Multiplicative non-identifiability.** Without additional restrictions, the factorization into baseline and learned modulation is not identifiable because \((R,A)\mapsto(Rc,A/c)\) leaves the complete intensity unchanged.
- **Likelihood and information.** Conditional score and Fisher-information identities are derived for identified parametric modulation families.
- **Law-level stability.** A common Poisson-random-measure coupling gives quantitative total-variation control in terms of integrated intensity perturbations.
- **Approximation in law.** Uniform approximation of a target modulation function transfers to approximation of the induced Cox-process law.
- **Long-run theory.** The paper establishes compensator laws of large numbers and a random-compensator central limit theorem.

The computational experiments are designed to illustrate these theoretical results rather than to serve only as a generic prediction benchmark.

---

## Repository contents

The principal files currently included are:

| File | Description |
|---|---|
| `aim_dspp_simulation_colab.py` | Main Python implementation of the simulation study. |
| `aim_dspp_simulation_colab_py.ipynb` | Colab/Jupyter notebook version of the simulation workflow. |
| `AIM_DSPP_Simulation_Code.zip` | Archived simulation code. |
| `AIM_DSPP_Simulation_Outputs.zip` | Simulation figures, tables, and numerical outputs used for manuscript verification. |
| `AIM_DSPP_RealBiology_Colab_CORRECTED.py` | Corrected Python implementation of the biological-data analyses. |
| `AIM_DSPP_RealBiology_Outputs_REPAIRED.zip` | Recommended corrected/repaired output archive for the biological applications. |
| `AIM_DSPP_RealBiology_Outputs.zip` | Earlier archived biological-analysis outputs retained for provenance. |
| `LICENSE` | MIT License. |

For reproduction of the final biological results, use the **corrected script** together with the **`REPAIRED` output archive**.

---

## Simulation study

The simulation study examines seven aspects of the theory:

1. conditional-Poisson calibration under the true directing intensity;
2. Cox overdispersion;
3. multiplicative non-identifiability;
4. held-out estimation of nonlinear exogenous modulation;
5. random-time-rescaling diagnostics;
6. common-Poisson-random-measure stability under score perturbations;
7. neural approximation and long-run compensator behavior.

The principal finite-sample comparison includes:

- Cox/directing baseline only;
- Linear AIM;
- Neural AIM-DSPP;
- Oracle intensity, used only as a reference.

The reported simulation results should not be interpreted as a claim that a neural modulator must dominate every alternative. The oracle is not a fitted competitor, and the theoretical results concern the stochastic-process structure rather than a universal ranking of estimators.

### Main simulation findings

Among the reported results:

- the true AIM-DSPP construction gives near-ideal conditional-Poisson diagnostics;
- strong marginal overdispersion coexists with conditional Poisson behavior;
- distinct baseline/modulation factorizations can generate the same intensity and likelihood to numerical precision;
- on the simulated nonlinear problem, the neural modulator performs close to the oracle and substantially improves intensity recovery relative to lower-capacity fitted baselines;
- the common-PRM mismatch experiment follows the theoretical coupling expression closely;
- function-approximation error transfers to small law-level discrepancy as predicted by the theory;
- the compensator LLN is already pronounced at moderate horizons, while finite-horizon normality remains more demanding.

---

## Real biological applications

Two public repeated-count datasets are analyzed as empirical projections of the multiplicative structure

\[
\mu_i=R_iA_\theta(X_i).
\]

Exact within-window event times are **not fabricated**. Accordingly, these applications assess the multiplicative conditional-mean architecture and discrete calibration diagnostics rather than the continuous-time time-rescaling theorem.

### 1. Epileptic seizure counts

The analysis uses the `epil` dataset distributed with the R package `MASS`, originating from the longitudinal seizure-count study of Thall and Vail.

The directing baseline is the observed pretreatment seizure rate:

\[
R_i=\frac{B_i}{4},
\]

where \(B_i\) is the eight-week pretreatment seizure count.

Patients are split at the **patient level**, so held-out subjects do not appear in training or validation.

A scientifically important result is that the directing baseline alone has the best held-out NLL in this application. Learned modulation does **not** improve prediction on unseen patients. This is retained deliberately as a contrasting empirical regime and demonstrates that the proposed framework is not presented as a “neural method always wins” procedure.

Dataset documentation:

https://search.r-project.org/CRAN/refmans/MASS/html/epil.html

### 2. Barn-owl begging counts

The analysis uses the `Owls` dataset distributed with `glmmTMB`, associated with the barn-owl experiment of Roulin and Bersier.

A history-only block within each nest is used to construct a shrunk nest-specific directing rate. Subsequent non-history bouts are then modeled using the baseline

\[
R_{jk}=B_{jk}\widehat r_j,
\]

where \(B_{jk}\) is brood size.

In this application, exogenous modulation materially improves the history-only directing baseline. The spline AIM has the smallest point-estimate test NLL, while the neural AIM-DSPP also improves substantially over the directing baseline. Cluster-bootstrap comparisons support the value of **modulation itself**, but do not establish universal superiority of the neural architecture over the linear or spline alternatives.

Dataset documentation:

https://glmmtmb.github.io/glmmTMB/reference/Owls.html

Official data object:

https://raw.githubusercontent.com/glmmTMB/glmmTMB/master/glmmTMB/data/Owls.rda

---

## Reproducibility

The code is written for Python and is designed to run conveniently in **Google Colab** or a standard Python environment.

A typical workflow is:

```bash
git clone https://github.com/debashisdotchatterjee/aim_dspp_2026.git
cd aim_dspp_2026
```

Then run the simulation study:

```bash
python aim_dspp_simulation_colab.py
```

and the corrected biological-data analysis:

```bash
python AIM_DSPP_RealBiology_Colab_CORRECTED.py
```

The scripts install or import the scientific Python packages required by the respective workflows. Exact package requirements can be read directly from the scripts/notebook.

Because stochastic optimization and Monte Carlo procedures are used, the analysis employs fixed random seeds where specified in the scripts and manuscript.

---

## Important interpretation notes

### Cox process versus history-dependent neural point process

A random or neural conditional intensity is not automatically a Cox process.

The AIM-DSPP construction requires that the baseline, exogenous features, and learned randomization be included in the directing information **before** the Poisson innovation is generated. If the same event history being generated is recursively used to construct the intensity, the resulting model is generally a history-dependent conditional-intensity process and requires a separate argument to admit a Cox representation.

### Identifiability

Only the product \(RA\) is generally identified from event data without further restrictions. The individual factors should therefore not automatically be given separate scientific or causal interpretations.

### Empirical calibration

The biological analyses use plug-in directing components. Residual overdispersion and randomized-PIT departures should therefore be interpreted as limitations of the plug-in conditional-Poisson approximation, not as a general rejection of Cox-process modeling.

---

## Data availability

No new biological data were collected for this study.

The empirical analyses use publicly available datasets:

- **Epilepsy:** `MASS::epil`
- **Barn owls:** `glmmTMB::Owls`

All simulation data are synthetic and can be regenerated from the code in this repository.

---

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@article{ChatterjeeAIMDSPP2026,
  author  = {Debashis Chatterjee},
  title   = {Cox Processes with Exogenous AI Modulation:
             Identifiability, Stability, and Approximation in Law},
  year    = {2026},
  note    = {Manuscript}
}
```

The bibliographic entry should be updated with the journal, volume, pages, and DOI after publication.

---

## Author

**Dr. Debashis Chatterjee**  
Department of Statistics  
Visva-Bharati University  
Santiniketan, West Bengal, India

Email: `debashis.chatterjee@visva-bharati.ac.in`

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Repository

https://github.com/debashisdotchatterjee/aim_dspp_2026
