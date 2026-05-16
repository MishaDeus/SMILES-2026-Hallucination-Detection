\# SOLUTION.md



\## 1. Reproducibility Instructions



All experiments were run in Google Colab T4 GPU; All random seeds were fixed with random\_state=42. To reproduce the results you may use default commands:



git clone https://github.com/MishaDeus/SMILES-2026-Hallucination-Detection.git



cd SMILES-2026-Hallucination-Detection



pip install -r requirements.txt



python solution.py



\---



\## 2. Final Solution Description



The final approach was initially based on the extraction of hidden states from the 13th, 14th and 15th layers of the Qwen2.5-0.5B model, with slight adjustments, which are described below.

Focus was placed on using last tokens of input tensor "hidden\_states".

\### Key Components



\*   Response-Focused Pooling: The last 19 tokens were extracted. Usage of last 19 tokens ensures that only assistans' response is used for aggregation, and prompt is ignored.

\*   Mean with Std Feature Engineering: For each sample, the mean vector and the standard deviation vector were concatenated.

&#x20;   \*   The Mean captures the semantic direction of the answer.

&#x20;   \*   The Std (Standard Deviation) acts as a proxy for "epistemic uncertainty" or activation volatility during a hallucination.

\*   Geometric Features: Layer-wise Activation Norms (L2) across all 24 layers to detect anomaly spikes and cosine similarity between adjacent layers to account for internal logic "drift". 



\### Classifier and Validation



\*   Robust Classifier: A RandomForestClassifier with limited depth (max\_depth=3, min\_samples\_leaf=15) was utilized to prevent overfitting, since dataset was very limited (689 examples).

\*   Validation Strategy: 5-fold Stratified Cross-Validation was implemented, providing a stable and objective metric evaluation.



It should be mentioned that RandomForestClassifier alone did not increase final scores. It worked in combination with Stratified Cross-Validation.



Additionally, 13th, 14th and 15th layers were selected because in general middle layers in LLMs are prone to hallucinations.



\---



\## 3. Experiments and Failed Attempts



\### 1. Global Mean Pooling (All Tokens)

\*   Result: \~64-66% AUROC.

\*   Possible reason for failure: Hidden states were "diluted" by the long and repetitive prompt text, washing out the specific signal associated with the assistant's hallucination.



\### 2. Logistic Regression (L1/L2)

\*   Result: High training AUROC (98%+) but poor test performance (\~68%).

\*   Possible reason for failure: Linear models were found to be too prone to memorizing specific noise in the high-dimensional space rather than identifying the underlying pattern of error.



\### 3. PCA

\*   Result: Significant drop in accuracy.

\*   Possible reason for failure: Subtle signals indicating a hallucination were often captured in lower-variance components, which PCA discarded as noise.



\### 4. Single Token vs. Window

\*   Result: Utilization of only the final token yielded only \~57% AUROC.

\### 5. Modifying aggregation\_and\_feature\_extraction

\*   Experimented with non-uniform layer weights , global representation drift, and concatenating features from the final 24th layer.

\*   Result: Slightly degraded performance (\~76.03%)

\* Possible reason for failure: Adding more high-dimensional features over-saturated the Random Forest classifier, causing it to overfit to the training folds.



\---



\## 4. Final Results



\*   Primary Metric (Test AUROC): 76.58%

\*   Conclusion: The majority-class baseline and initial linear probe attempts were slightly outperformed. Overusing different methods on a such small dataset leads to a performance decrease.

