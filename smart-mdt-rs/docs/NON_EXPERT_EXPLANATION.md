# Non-expert explanation

Classical decision trees ask one question at a time. Multivariate decision trees can ask a combined logical question involving several features. This can make the tree smaller, but explaining the decision becomes harder. The paper shows that if the logical questions belong to certain tractable languages, we can still compute abductive explanations efficiently. This implementation builds trees using those safe languages and stores certificates showing which explanation backend is valid.

This project implements a certificate-guided Rust learner for compact and explainable multivariate decision trees. The certified modes restrict split conditions to tractable logical families and compute subset-minimal abductive explanations by deletion-based weak AXp checking.

The project does not prove the learner is optimal, does not prove all learned trees are globally best, does not certify arbitrary mixed-language paths, and does not formally verify the Rust implementation. Boolean affine paths and `smart_certified` trees whose every path passes compatibility validation are theorem-table eligible; tuned and empirical modes remain excluded.
