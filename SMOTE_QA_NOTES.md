# Ordinal-SMOTE: Question & Answer Notes
## For Research Report — Detailed Reference

---

## Q1: After Ordinal-SMOTE, 12,000 dataset theke 29,679 kivabe holo?

### Answer:

Original 12,000 data delete hoyni. SMOTE shudhu **notun synthetic samples ADD kore** — kono real sample remove kore na.

### Step-by-Step Breakdown:

**Original Dataset (Before SMOTE):**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 510 | 4.3% |
| Pre-Diabetic (1) | 1,597 | 13.3% |
| Diabetic (2) | 9,893 | 82.4% |
| **Total** | **12,000** | **100%** |

---

**Ordinal-SMOTE Step 1 — Normal ↔ Pre-Diabetic (N↔P) Balance:**
- Normal (510) ke Pre-Diabetic (1,597) er soman korte
- ~1,087 ta synthetic Normal sample ADD holo
- Normal ekhon → 1,597

**Ordinal-SMOTE Step 2 — Pre-Diabetic ↔ Diabetic (P↔Y) Balance:**
- Pre-Diabetic (1,597) ke Diabetic (9,893) er soman korte
- ~8,296 ta synthetic Pre-Diabetic sample ADD holo
- Pre-Diabetic ekhon → 9,893

**Ordinal-SMOTE Step 3 — Normal Final Balance:**
- Normal (1,597) ekhono Diabetic (9,893) er tulonay onek kom
- Aaro ~8,296 ta synthetic Normal sample ADD holo
- Normal ekhon → 9,893

**Final Dataset (After SMOTE):**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 9,893 | 33.3% |
| Pre-Diabetic (1) | 9,893 | 33.3% |
| Diabetic (2) | 9,893 | 33.3% |
| **Total** | **29,679** | **100%** |

### Key Point:
> Original 12,000 samples intact ache. SMOTE shudhu minority class gulote **synthetic samples generate kore add koreche** — jotokhon na tinta class soman hoy. Tai 12,000 → 29,679.

---

## Q2: Synthetic Normal Sample ADD — eita kivabe kaj kore?

### Answer:

SMOTE aslে **duita real sample er majhkhane ekta notun sample toiri kore.**

Eta kono random data invention na — real patients er feature values er weighted interpolation.

---

### Mathematical Formula:

```
Synthetic Sample = Sample_A + λ × (Sample_B − Sample_A)

jekhane:
  Sample_A  = ekjon real patient er feature vector
  Sample_B  = tar nearest neighbor (same class er arekjon real patient)
  λ         = random number between 0 and 1
```

---

### Concrete Example:

Normal class e duijone real patient ache:

| Patient | HbA1c | FBS (mg/dL) | Age | BMI |
|---------|-------|-------------|-----|-----|
| Patient A | 5.2 | 88 | 35 | 22 |
| Patient B | 5.4 | 92 | 40 | 24 |

λ = 0.5 hole SMOTE ekta synthetic patient banabe:

| Feature | Calculation | Synthetic Value |
|---------|-------------|-----------------|
| HbA1c | 5.2 + 0.5 × (5.4 − 5.2) | **5.3** |
| FBS | 88 + 0.5 × (92 − 88) | **90** |
| Age | 35 + 0.5 × (40 − 35) | **37** |
| BMI | 22 + 0.5 × (24 − 22) | **23** |

Result:

| Synthetic Patient C | HbA1c | FBS | Age | BMI |
|--------------------|-------|-----|-----|-----|
| (New — Generated) | 5.3 | 90 | 37 | 23 |

---

### Ordinal-SMOTE er Extra Constraint:

Standard SMOTE te kono boundary nei — Normal er synthetic sample banate giye Diabetic er kacha sample banie felote pare.

Amader **Ordinal-SMOTE** e strict constraint ache:

| Rule | Description |
|------|-------------|
| N↔P only | Normal er synthetic sample banate shudhu Normal patients der modhye interpolate kore |
| P↔Y only | Pre-Diabetic er synthetic sample banate shudhu Pre-Diabetic patients der modhye interpolate kore |
| N↔Y never | Normal theke directly Diabetic er dike kono interpolation hoy na |

Ei constraint er karone ordinal boundary **N < P < Y** sorboda safe thake.

---

### Visualization of the Concept:

```
Feature Space (HbA1c axis example):

Normal Range:        [4.5 ──── A ──── C(synthetic) ──── B ──── 5.6]
                                      ↑
                              interpolated here
                              (between two Normal patients only)

Pre-Diabetic Range:  [5.7 ──────────────────────────────────── 6.4]

Diabetic Range:      [6.5 ──────────────────────────────────── 9.0+]

Ordinal-SMOTE NEVER crosses these boundaries.
```

---

### Ekta Kotha e Summary:

> SMOTE notun patient **invent** kore na — duijone real patient er feature values er **majhamajhi** ekta notun realistic patient toiri kore. Ordinal-SMOTE e additional rule holo — ei interpolation shudhu **same class er patients der modhye** hoy, jate ordinal order (Normal < Pre-Diabetic < Diabetic) kakhono vanga na pore.

---

## Report e Add Korar Jonno Suggested Paragraph:

*"To address the severe class imbalance in the dataset (Normal: 4.3%, Pre-Diabetic: 13.3%, Diabetic: 82.4%), a custom Ordinal-SMOTE algorithm was implemented. Unlike standard SMOTE, which generates synthetic samples without regard for class ordering, Ordinal-SMOTE restricts interpolation strictly to adjacent classes (N↔P and P↔Y), ensuring that the ordinal constraint N < P < Y is preserved in all synthetic samples. The algorithm operates by selecting two real samples from the same class and generating a new synthetic sample at a random interpolation point between them in feature space, following the formula: Synthetic = Sample_A + λ × (Sample_B − Sample_A), where λ ∈ [0, 1]. This process was applied iteratively until all three classes reached equal representation of 9,893 samples each, resulting in a balanced dataset of 29,679 total samples."*

---

*Notes prepared for research report reference.*
*Based on implementation in: data_preprocessing.py → OrdinalSMOTE class*
