use std::fmt;

/// Logical family of a split/path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LanguageFamily {
    Unary,
    Horn,
    AntiHorn,
    Square2Cnf,
    /// Certified Boolean affine (single GF(2) linear equation over Boolean variables).
    Affine,
    /// Certified policy whose individual paths may use different tractable families.
    SmartCertified,
    EmpiricalAffine,
    EmpiricalMixed,
    TunedExperimental,
}
impl LanguageFamily {
    /// True when allowed in theorem-certified result tables.
    pub fn theorem_table_allowed(self) -> bool {
        matches!(
            self,
            Self::Unary
                | Self::Horn
                | Self::AntiHorn
                | Self::Square2Cnf
                | Self::Affine
                | Self::SmartCertified
        )
    }
}

/// Certified predicate families that an optimizer may generate.
///
/// This mask is deliberately independent of the optimizer profile: CALS and
/// CompactExplain keep all of their search, scoring, cache, and pruning
/// settings while this value changes only the admissible candidate families.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllowedLanguages(u8);

impl AllowedLanguages {
    const UNARY: u8 = 1 << 0;
    const HORN: u8 = 1 << 1;
    const ANTIHORN: u8 = 1 << 2;
    const SQUARE2CNF: u8 = 1 << 3;
    const AFFINE: u8 = 1 << 4;
    const ALL: u8 = Self::UNARY | Self::HORN | Self::ANTIHORN | Self::SQUARE2CNF | Self::AFFINE;

    pub const fn all() -> Self {
        Self(Self::ALL)
    }

    pub const fn only(family: LanguageFamily) -> Self {
        Self(match family {
            LanguageFamily::Unary => Self::UNARY,
            LanguageFamily::Horn => Self::HORN,
            LanguageFamily::AntiHorn => Self::ANTIHORN,
            LanguageFamily::Square2Cnf => Self::SQUARE2CNF,
            LanguageFamily::Affine => Self::AFFINE,
            _ => 0,
        })
    }

    pub const fn contains(self, family: LanguageFamily) -> bool {
        let bit = match family {
            LanguageFamily::Unary => Self::UNARY,
            LanguageFamily::Horn => Self::HORN,
            LanguageFamily::AntiHorn => Self::ANTIHORN,
            LanguageFamily::Square2Cnf => Self::SQUARE2CNF,
            LanguageFamily::Affine => Self::AFFINE,
            _ => 0,
        };
        bit != 0 && self.0 & bit != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }
}

impl Default for AllowedLanguages {
    fn default() -> Self {
        Self::all()
    }
}

impl fmt::Display for AllowedLanguages {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let families = [
            (LanguageFamily::Unary, "Unary"),
            (LanguageFamily::Horn, "Horn"),
            (LanguageFamily::AntiHorn, "AntiHorn"),
            (LanguageFamily::Square2Cnf, "Square2CNF"),
            (LanguageFamily::Affine, "Affine"),
        ];
        let mut separator = "";
        for (family, name) in families {
            if self.contains(family) {
                formatter.write_str(separator)?;
                formatter.write_str(name)?;
                separator = "|";
            }
        }
        Ok(())
    }
}

impl fmt::Debug for AllowedLanguages {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("AllowedLanguages")
            .field(&self.to_string())
            .finish()
    }
}
