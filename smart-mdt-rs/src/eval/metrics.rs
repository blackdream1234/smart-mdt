use crate::ClassId;
/// Classification accuracy.
pub fn accuracy(y: &[ClassId], p: &[ClassId]) -> f64 {
    if y.is_empty() {
        0.0
    } else {
        y.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / y.len() as f64
    }
}
