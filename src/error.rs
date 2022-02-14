
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Exceeded the depth limit of {max_depth:?} levels for this Morton index type")]
    DepthLimitedExceeded{
        max_depth: usize,
    },
    #[error("Cell index is out of range")]
    CellIndexOutOfRange,
}