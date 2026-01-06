# Method Details: Robotic Food Cutting Holding Point Search

This document details the algorithm described in the paper "Searching Method for Holding Points during Robotic Food Cutting" (Dong et al., SICE 2025), specifically Chapters III, IV, and V. It also reviews the system requirements document to ensure alignment and identify potential issues.

---

## Part 1: Detailed Method Description (From Paper)

### 1. Problem Formulation (Section III.A)
- **Inputs:**
  - Food surface shape $\Omega$ (Point cloud/mesh).
  - Knife trajectory $\mathbb{T} = \{(T_1, v_1), (T_2, v_2), \dots\}$, where $T$ is pose and $v$ is velocity.
  - Number of fingers $N$ (usually 2).
  - Physical parameters (friction coefficient $\mu$, fracture toughness $\kappa$, etc.).
- **Output:**
  - An optimal holding point set $P = \{p_1, \dots, p_n\}$ where $p_i$ is the position of the $i$-th finger.
- **Assumptions:**
  - Motion: Translation in XY, Rotation about Z.
  - Stable planar contact between food and cutting board.
  - Point contact (fingers), only pushing force and friction; no torsional friction.
  - Center of mass $C_o$ is known.
  - Planar cutting (knife stays in a plane).

### 2. Overall Workflow (Algorithm 1: Full Progress)
The method is an offline search over a trajectory.

1.  **Initialization:**
    - Generate all possible holding point sets $\mathbb{P}_{all}$ from the food surface part $\Omega_g$ (holding part).
    - Initialize a score dictionary $\mathbb{S}$ for all $P \in \mathbb{P}_{all}$ with 0.
2.  **Trajectory Loop:** For each time step $(T, v) \in \mathbb{T}$:
    - **Data Preparation (`PREPAREDATA`):**
        - Calculate contact surfaces $\Omega_{c1}, \Omega_{c2}$ (where knife touches food) and holding part $\Omega_g$.
        - Identify valid initial candidate sets $\mathbb{P}_{init}$ that reside on $\Omega_g$.
    - **Geometric Filtering (`FILTERBYGEOSCORE` - Algorithm 2):**
        - Calculate geometric scores for candidates in $\mathbb{P}_{init}$.
        - Keep only the top portion (ratio $r$) of candidates; let this filtered set be $\mathbb{P}$.
        - **Critical Step:** For any $P \in (\mathbb{P}_{all} \setminus \mathbb{P})$, set $\mathbb{S}[P] \leftarrow -\infty$. (Once eliminated, never recovers).
    - **Positional Scoring (`CALPOSITIONALSCORE` - Algorithm 3):**
        - Calculate $S_{pos}$ for remaining $P \in \mathbb{P}$.
    - **Knife Wrench Calculation (`CALKNIFEFORCE`):**
        - Calculate total force/torque $t_k$ exerted by the knife on the food.
    - **Dynamic Scoring (`CALDYNAMICSCORE` - Algorithm 4):**
        - Calculate $S_{force}$ for remaining $P \in \mathbb{P}$ by generating feasible holding forces to resist $t_k$.
    - **Accumulation:**
        - $\mathbb{S}[P] \leftarrow \mathbb{S}[P] + S_{pos} + S_{force}$ for valid $P$.
3.  **Selection:**
    - Return $P_{fin} = \arg\max_{P} \mathbb{S}[P]$.

### 3. Search Space Reduction via Geometric Constraints (Section IV, Algorithm 2)
To reduce computational cost, candidates are filtered using geometry before expensive force calculations.

- **Scores calculated per candidate $P$:**
    1.  **Finger Spacing ($E_{fin}$):** Penalize fingers being too close (acting as one).
        $$D(x) = \begin{cases} \frac{e^{ax/b}-1}{e^a-1} & (x \le b) \\ 1 & (\text{otherwise}) \end{cases}$$ 
        $$E_{fin} = \min_{i \ne j} D(\|p_i - p_j\|)$$
        *($b$ is upper limit distance, $a$ adjusts curvature)*
    2.  **Knife Proximity ($E_{knf}$):** Penalize fingers close to the knife (collision risk).
        $$E_{knf} = \min_{i} (\text{Dist}(p_i, \hat{n}_k, p_k))$$ 
        *($\hat{n}_k$ is knife normal, $p_k$ is knife position)*
    3.  **Table Collision ($E_{tbl}$):** Penalize fingers close to the table (low z-height).
        $$E_{tbl} = \min_{i} (p_{i,z})$$
- **Normalization & Combination:**
    - Normalize each component to $[0, 1]$ across the current candidate set.
    - $S_{geo} = w_{fin}E_{fin} + w_{knf}E_{knf} + w_{tbl}E_{tbl}$.
- **Filtering:**
    - Sort $\mathbb{P}_{init}$ by $S_{geo}$.
    - Return top $\lfloor r \cdot N_{elements} \rfloor$ candidates.

### 4. Evaluation Score Calculation (Section V)

#### A. Positional Score ($S_{pos}$, Algorithm 3)
Evaluates stability based on position relative to knife.
- **Direction Score ($E_{pdir}$):** Prefer finger arrangements parallel to knife plane (better resistance to torque/force).
    $$E_{pdir} = 1 - \|PCA(P) \cdot \hat{n}_k\|$$ 
- **Distance Score ($E_{pdis}$):** Prefer fingers closer to knife plane (shorter moment arm).
    $$E_{pdis} = -\min(\text{Dist}(P, \hat{n}_k, p_k))$$ 
- **Combination:**
    - Normalize components to $[0, 1]$.
    - $S_{pos} = w_{pdir}E_{pdir} + w_{pdis}E_{pdis}$.

#### B. Knife Wrench Calculation ($t_k$)
Decomposed into Fracture Force and Friction Force.
1.  **Fracture Force ($f_c, \tau_c$):**
    - $f_c(u) = -\kappa (\hat{v}(u) \cdot \hat{n}_s(u)) \hat{v}(u)$ (integrated along blade curve).
    - $\tau_c = \int (s(u) - g) \times f_c(u) du$.
2.  **Friction Force ($f_{fr}, \tau_{fr}$):**
    - Assumes uniform pressure $P$ on contact surfaces $\Omega_{c1}, \Omega_{c2}$.
    - $f_{fr} = \mu P \sum \iint \hat{v}_{cj} dS$ (Coulomb friction).
    - $\tau_{fr} = \mu P \sum \iint (x - g) \times \hat{v}_{cj} dS$.
3.  **Total Wrench:** $t_k = [(f_{fr}+f_c)^T, (\tau_{fr}+\tau_c)^T]^T$.

#### C. Dynamic Score ($S_{force}$, Algorithm 4)
Evaluates ability to generate required holding force $f$ such that resultant holding wrench $t_h = -t_k$.
- **Relation:** $t_h = Gf$, where $G$ is the grasping matrix.
- **Force Generation Method (Proposed):**
    1.  Randomly generate $f_{init}$ inside friction cone.
    2.  Calculate residual needed: $t = (-t_k) - G f_{init}$.
    3.  Adjust: $f = f_{init} - G^+ t$ ($G^+$ is pseudo-inverse).
    4.  Check if $f$ is within friction cone.
- **Scoring (for a valid $f$):**
    1.  **Magnitude ($E_{mag}$):** Prefer smaller forces. $E_{mag} = -\sum \|f_{fi}\|$. 
    2.  **Direction ($E_{dir}$):** Prefer forces pushing *into* food (normal alignment). $E_{dir} = \sum \hat{n}_\Omega(p_i) \cdot \hat{f}_{fi}$.
    3.  **Variance ($E_{var}$):** Prefer uniform load distribution. $E_{var} = -\text{Var}(\|f_{fi}\|)$.
- **Selection:**
    - Generate $M$ trial forces for a candidate set $P$.
    - Calculate score $S = w_{mag}E_{mag} + w_{dir}E_{dir} + w_{var}E_{var}$ (normalized).
    - $S_{force} = \max_j (S_j)$.

---

## Part 2: Review against Requirements Document (Rev 2)

### 1. Alignment Check
The Requirements Document (Rev 2) generally aligns very well with the paper's logic.
- **Architecture:** The Python/C++ split is suitable for the heavy geometric/numeric computations.
- **Algorithms:** Explicitly maps Algorithm 1-4.
- **Scoring:** Formulas match.
- **Wrench:** The abstraction of `ContactSurfaceData` is a good engineering decision to separate geometry from physics integration.

### 2. Discrepancies & Clarifications

#### A. Candidate Generation
- **Paper:** "Randomly generate many holding point set... collect into $P_{all}$".
- **Requirements:** "Generate all $C_M^N$ combinations" (Exhaustive).
- **Analysis:** This is a **positive deviation**. For $N=2$ and $M \approx 100$ (downsampled), $C_{100}^2 = 4950$, which is trivial to compute exhaustively. This removes randomness from the search space definition, improving reproducibility as requested ("可复现").

#### B. Geometric Filtering "Cut-off"
- **Paper:** Algorithm 2 returns "top $\lfloor rN \rfloor$".
- **Requirements:** Uses `geo_filter_ratio` ($r$).
- **Note:** The requirements adhere to the paper. The critical behavior "Once marked $-\infty$, never recover" is correctly captured as an invariant. This is crucial: if a hand position is blocked by the knife at *any* timestep (even just one), that hand position is invalid for the *entire* task.

#### C. Normalization Context
- **Paper:** implies normalization is performed on the *current* set of candidates being evaluated in that function.
- **Requirements:** "Normalization (min-max) only used for relative sorting of current time step candidates, does not change cross-time accumulation semantics."
- **Analysis:** This is correct. You cannot normalize the *accumulated* score at every step, or you destroy the history. You normalize the *step* score ($S_{pos}, S_{force}$) to [0,1] before adding it to the accumulator.

### 3. Potential Process Errors / Unclear Points

1.  **Random Seed Control (Reproducibility):**
    - The requirements stress "Reproducible" ("可复现").
    - The paper's force generation (Algorithm 4) and initial search space (if random) rely on RNG.
    - **Action Item:** The implementation *must* allow passing a fixed random seed to the C++ `ScoreCalculator` or specific methods (`calcDynamicScores`). The config should have a field for `random_seed`.

2.  **Handling "No Feasible Force":**
    - The paper doesn't explicitly state what happens if *no* generated force satisfies the friction cone in Algorithm 4 (i.e., all $M$ trials fail).
    - **Requirements:** Explicitly states "Physical failure... -> $S = -\infty$".
    - **Validation:** This is the correct logical deduction. If you can't hold it, the score is negative infinity.

3.  **Integration Precision vs Speed:**
    - The paper uses integrals for fracture/friction.
    - **Requirements:** "C++... fracture/friction integral".
    - **Detail:** The discretization resolution for these integrals needs to be defined (e.g., step size along blade, grid size on contact surface). This is hidden in "ContactSurfaceData" generation.

4.  **$\Omega_g$ Definition:**
    - Paper: $\Omega_g$ is the "holding part".
    - Requirements: $\Omega_g$ is realized as a list of `valid_indices`.
    - **Clarification:** The paper mentions $\Omega_g$ is determined by dividing food by knife plane. In practice, points "behind" the knife (relative to cutting direction) or on the larger side are valid. The implementation of `get_valid_indices` in Python needs to be robust (e.g., dot product with knife normal > 0, or explicit mesh cutting).

### 4. Summary for Implementation
The requirements document is solid. The main area for caution is the **Force Generation Loop** in C++:
- It requires efficient implementation (batching/OpenMP).
- It must handle the "zero valid forces" case gracefully (return $-\infty$).
- It must use a deterministic RNG state if reproducibility is required.
