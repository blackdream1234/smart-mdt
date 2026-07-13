use super::{Backend, Predicate};
use crate::{Result, SmartMdtError};

/// Tractable theory committed to by one root-to-current-node path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PathTheoryState {
    Uncommitted,
    Horn,
    AntiHorn,
    TwoSat,
    AffineGf2,
}

impl PathTheoryState {
    /// Stable value used in benchmark and debug metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uncommitted => "uncommitted",
            Self::Horn => "horn",
            Self::AntiHorn => "antihorn",
            Self::TwoSat => "two_sat",
            Self::AffineGf2 => "affine_gf2",
        }
    }

    /// Certified backend for a completed path in this state.
    ///
    /// An uncommitted path contains unary predicates only, which use the
    /// structural Horn backend.
    pub fn backend(self) -> Backend {
        match self {
            Self::Uncommitted | Self::Horn => Backend::StructuralHorn,
            Self::AntiHorn => Backend::StructuralAntiHorn,
            Self::TwoSat => Backend::TwoSat,
            Self::AffineGf2 => Backend::Gf2Gaussian,
        }
    }
}

/// Returns whether `predicate` can extend a path in `state` without mixing
/// incompatible tractable languages.
pub fn candidate_is_compatible(state: PathTheoryState, predicate: &Predicate) -> bool {
    match predicate {
        Predicate::Unary(_) => true,
        Predicate::HornClause(_) => {
            matches!(state, PathTheoryState::Uncommitted | PathTheoryState::Horn)
        }
        Predicate::AntiHornClause(_) => matches!(
            state,
            PathTheoryState::Uncommitted | PathTheoryState::AntiHorn
        ),
        Predicate::Square2Cnf { .. } => matches!(
            state,
            PathTheoryState::Uncommitted | PathTheoryState::TwoSat
        ),
        Predicate::Affine { .. } => matches!(
            state,
            PathTheoryState::Uncommitted | PathTheoryState::AffineGf2
        ),
        Predicate::EmpiricalAffine { .. } => false,
    }
}

/// Advances the tractable theory for a path, rejecting incompatible predicates.
pub fn next_theory_state(state: PathTheoryState, predicate: &Predicate) -> Result<PathTheoryState> {
    if !candidate_is_compatible(state, predicate) {
        return Err(SmartMdtError::TheoremRejected(format!(
            "predicate family {:?} is incompatible with path theory {:?}",
            predicate.language(),
            state
        )));
    }
    Ok(match predicate {
        Predicate::Unary(_) => state,
        Predicate::HornClause(_) => PathTheoryState::Horn,
        Predicate::AntiHornClause(_) => PathTheoryState::AntiHorn,
        Predicate::Square2Cnf { .. } => PathTheoryState::TwoSat,
        Predicate::Affine { .. } => PathTheoryState::AffineGf2,
        Predicate::EmpiricalAffine { .. } => unreachable!("rejected above"),
    })
}
