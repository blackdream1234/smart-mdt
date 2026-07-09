/// Logical family of a split/path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LanguageFamily {
    Unary,
    Horn,
    AntiHorn,
    Square2Cnf,
    /// Certified Boolean affine (single GF(2) linear equation over Boolean variables).
    Affine,
    EmpiricalAffine,
    EmpiricalMixed,
    TunedExperimental,
}
impl LanguageFamily {
    /// True when allowed in theorem-certified result tables.
    pub fn theorem_table_allowed(self) -> bool {
        matches!(
            self,
            Self::Unary | Self::Horn | Self::AntiHorn | Self::Square2Cnf | Self::Affine
        )
    }
}
