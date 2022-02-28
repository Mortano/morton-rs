
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Exceeded the depth limit of {max_depth:?} levels for this Morton index type")]
    DepthLimitedExceeded{
        max_depth: usize,
    },
    #[error("Cell index is out of range")]
    CellIndexOutOfRange,
}

pub type Result<T> = std::result::Result<T, crate::Error>;