/// Logical family of a split/path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LanguageFamily {
    Unary,
    Horn,
    AntiHorn,
    Square2Cnf,
    EmpiricalAffine,
    EmpiricalMixed,
    TunedExperimental,
}
impl LanguageFamily {
    /// True when allowed in theorem-certified result tables.
    pub fn theorem_table_allowed(self) -> bool {
        matches!(
            self,
            Self::Unary | Self::Horn | Self::AntiHorn | Self::Square2Cnf
        )
    }
}
