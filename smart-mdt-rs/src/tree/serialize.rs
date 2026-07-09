use super::TreeNode;
use crate::Result;
/// Serializes a tree as debug text placeholder.
pub fn to_json(tree: &TreeNode) -> Result<String> {
    Ok(format!("{tree:#?}"))
}
/// Deserialization is reserved for the JSON phase.
pub fn from_json(_s: &str) -> Result<TreeNode> {
    Err(crate::SmartMdtError::Json(
        "model loading requires JSON feature phase".into(),
    ))
}
