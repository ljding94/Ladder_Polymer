# Ladder_Polymer

This project simulates the behavior of ladder polymers using Monte Carlo methods and Gaussian Process Regression (GPR) for data analysis. The sections below describe the polymer model and outline the workflow for generating simulation data, training the GPR, and applying experimental data.

Note: This code is intended primarily for the author's own use and may not be very user-friendly. For assistance or questions regarding its usage please contact the author.


## Requirements

The following packages are required to run the analysis code:

### Python Packages
- **numpy 2.1.2**: For numerical operations and array handling
- **matplotlib 3.9.2**: For data visualization and plotting
- **scikit-learn 1.5.2**: For Gaussian Process Regression implementation
- **scipy 1.14.1**: For scientific calculations
- **pandas 2.2.3**: For data manipulation and handling

## Model

Each monomer unit of the polymer is modeled as a rectangular segment with an orientation defined by two unit vectors:
- $\hat{u}$ : along the segment’s tangent direction (the "long" axis)
- $\hat{v}$ : along the short axis, perpendicular to $\hat{u}$

A polymer is represented as a chain of \(L\) segments, where \(L\) is the contour length in units of the monomer length \(B\).

To incorporate inherent bending and twisting, we introduce two additional unit vectors, $\hat{u}'$ and $\hat{v}'$, which specify the preferred orientation of the subsequent segment. For two connected segments \(i\) and \(j\), the angle between the pairs $(\hat{u}, \hat{v})$ and $(\hat{u}', \hat{v}')$ characterizes the inherent bending and twisting. In this model, we assume that $\hat{v} = \hat{v}'$, and the inherent bending is given by:

$$
\cos(\alpha) = \hat{u} \cdot \hat{u}'
$$

An energy cost is incurred when the orientation of segment \(i+1\), \((\hat{u}_{i+1},\hat{v}_{i+1})\), deviates from the preferred orientation of segment \(i\), \((\hat{u}'_i,\hat{v}'_i)\). The polymer energy is defined as:

$$
E = \sum_{i} \left( \frac{1}{2}K_t \phi_{i}^2 + \frac{1}{2}K_b \theta_{i}^2 \right),
$$

where:
- **Bending:** $\cos(\theta_{i}) = \hat{u}'_i \cdot \hat{u}_{i+1}$
- **Twisting:** $\cos(\phi_{i}) = \hat{v}'_i \cdot \hat{v}_{i+1}$
- **\(K_t\) and \(K_b\):** twisting and bending moduli, respectively.

Additionally, the preferred orientation for successive segments may not always be maintained on the same side. The probability of a link being an *anti-link* is defined by the anti rate, \(R_a\).

## Workflow

### 1. Generate Training Data via Monte Carlo Simulation

The simulation code is located in the `/code` directory. Follow these steps:

- **Compilation:**
  Use the provided Makefile to compile the simulation program.

- **Running the Simulation:**
  Execute the compiled binary with one of the following parameter sets:

  - **Randomly Generated Polymer Parameters:**
    ```bash
    ./biaxial_polymer segment_type run_num folder
    ```

  - **Specified Polymer Parameters:**
    ```bash
    ./biaxial_polymer segment_type lnLmu lnLsig Kt Kb Rf abs_alpha folder
    ```

  **Parameter Descriptions:**
  - `segment_type`: Either `inplane` or `outofplane` (determines the inherent bending direction).
  - `lnLmu` and `lnLsig`: Parameters for the polydisperse polymer system.
  - `Kt` and `Kb`: Twisting and bending moduli.
  - `Rf`: Probability of an anti-bond in the polymer chain.
  - `abs_alpha`: The inherent bending angle.

### 2. Train the Gaussian Process Regressor (GPR)

The GPR implementation is in the `analyze` folder. To train the model:

- **Run the Analysis Script:**
  ```bash
  python main_ML_analyze.py
  ```
This script trains the GPR on the simulation training set and evaluates its performance on the test set.

### 3. Apply Experimental Data

Prepare and process your experimental scattering data as follows:

- **Renormalization:**
  Normalize the scattering data to 1 at low \(q\) by fitting the Guinier region.

- **Rebinning:**
  Rebin the data to match the \(q\) values used in the simulation. This step can be performed using the SASview software.

After preprocessing, the experimental data can be fed into the GPR model in the same manner as the simulation test data.


