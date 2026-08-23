# Thesis Claims and Evidence Register

This register constrains thesis wording to what the exact theorem audit,
controlled experiment, and 46-dataset study actually support. `ESTABLISHED`
means established within the stated formal or experimental scope; it never
means that the program proves the paper's theorem or that a corpus result is a
universal law.

| # | Claim | Evidence source | Evidence type | Strength | Allowed wording | Forbidden wording |
|---:|---|---|---|---|---|---|
| 1 | The frozen benchmark respects the exact tractability theorem classes. | `theorem_reaudit.csv`; exact theorem tests; independent certification audit | Exhaustive artifact audit plus regression tests | **ESTABLISHED** | “All 7,360 frozen benchmark rows passed the exact theorem reaudit.” | “The program proves the theorem.” |
| 2 | Current learned Horn nodes belong to star-nested Horn. | Historical generator invariant at `1fbfa30`; `exact_theorem_classes.rs` | Structural/code audit | **ESTABLISHED** | “The current Horn learner emits single two-literal Horn clauses, a theorem-safe star-nested sublanguage.” | “The learner searches all possible star-nested Horn formulas.” |
| 3 | Current learned Anti-Horn nodes belong to star-nested Anti-Horn. | Historical generator invariant at `1fbfa30`; exact polarity-dual tests | Structural/code audit | **ESTABLISHED** | “The current Anti-Horn learner emits single two-literal Anti-Horn clauses, a theorem-safe star-nested sublanguage.” | “The learner searches the complete maximal star-nested Anti-Horn class.” |
| 4 | The Square2CNF learner searches the complete maximal Square2CNF class. | Generator audit; Theorem-6 recognizer tests | Negative implementation result | **FALSE** | “The current learner emits Form-I Square2CNF predicates, a theorem-safe sublanguage of the maximal class.” | “The current learner searches Forms I–III exhaustively.” |
| 5 | Boolean Affine directly represents parity. | Controlled XOR3 rows; `controlled_compactness_summary.csv` | Controlled exact-target experiment | **ESTABLISHED** | “Affine represented XOR3 perfectly with 3 nodes; Square2CNF and Unary also reached perfect accuracy but used 15 and 23 nodes.” | “Only Affine can represent XOR3.” |
| 6 | One fixed language is universally best. | `language_winners.csv`; `language_ranks.csv` | Dataset-level descriptive comparison | **FALSE** | “No fixed family was best on every evaluated dataset.” | “Unary/Horn/Square2CNF is universally best.” |
| 7 | Unary has the highest fixed-family win count. | `language_winners.csv` | Descriptive, deterministic tie-breaking | **DESCRIPTIVE** | “Unary had 19 wins under deterministic tie-breaking; the mean winner/runner-up margin was 0.00239.” | “Unary was substantially superior because it won most datasets.” |
| 8 | Horn, Anti-Horn, Unary, and Square2CNF differ significantly in accuracy. | `language_statistics.csv` | Friedman plus paired Wilcoxon/Holm analysis on 46 dataset units | **NOT ESTABLISHED** | “No pairwise accuracy difference among those four remained significant after Holm correction.” | “Those four families have significantly different accuracy under the corrected tests.” |
| 9 | Affine has lower average real-data accuracy than the other fixed families. | `language_statistics.csv`; dataset-aggregated means | Paired inferential analysis | **ESTABLISHED** | “On this corpus, Affine was lower than Unary (Holm p=0.026986, rank-biserial 0.489), Horn (0.000213, 0.684), Anti-Horn (0.000273, 0.673), and Square2CNF (0.000342, 0.661).” | “Affine is universally less accurate.” |
| 10 | Root logical signal predicts the best language. | `language_specialization_summary.csv` | Exploratory Spearman/bootstrap analysis, 46 dataset units | **NOT ESTABLISHED** | “No tested root-signal association was established; every 10,000-resample bootstrap interval crossed zero.” | “Horn signal determines when Horn should be used.” |
| 11 | CALS actually uses multiple theorem families. | `cals_language_usage.csv`; CALS final-node counts | Exhaustive descriptive diagnostic over the frozen corpus | **ESTABLISHED** | “CALS used all five families: Unary 18.5%, Horn 29.4%, Anti-Horn 10.9%, Square2CNF 34.7%, and Affine 6.6% of recorded final internal nodes.” | “CALS uses every family equally” or “family use proves causal benefit.” |
| 12 | CALS always beats the best fixed language. | `oracle_vs_adaptive.csv` | Dataset-level oracle comparison | **FALSE** | “CALS beat every globally fixed family on 21 datasets, matched on 2, and underperformed on 23; mean regret was slightly negative.” | “CALS always beats the fixed-language oracle.” |
| 13 | CALS can outperform the globally fixed-language oracle. | `oracle_vs_adaptive.csv` | Descriptive result on the evaluated corpus | **DESCRIPTIVE** | “On 21 evaluated datasets, CALS exceeded the best single family chosen globally per dataset, consistent with local family selection.” | “The fixed-language oracle is an unattainable theoretical optimum for all mixed-language trees.” |
| 14 | Controlled experiments establish language-specific compactness. | `controlled_compactness_summary.csv`; node/literal/AXp heatmaps | Controlled exact-target experiment | **ESTABLISHED** | “For the tested synthetic targets, compactness exposed native structural representations that accuracy alone concealed.” | “The controlled compactness pattern is a universal real-data selection rule.” |

## Thesis-safe synthesis

The formal evidence establishes exact theorem eligibility for every frozen
row. The controlled evidence establishes specialization and compactness only
for the tested Boolean targets. The real-data winner counts and CALS usage are
descriptive for this corpus. The structural-signal analysis remains
exploratory and did not establish a predictor of the best family.
