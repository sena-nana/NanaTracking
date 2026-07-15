use core::fmt;

use crate::{
    types::{NanaTrackingResult, SessionId},
    validate::{ContractError, Validate},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcceptedFrame {
    pub missing_sequences: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamError {
    WrongSession,
    WrongGeneration { expected: u32, actual: u32 },
    DuplicateOrOutOfOrder { last: u64, actual: u64 },
    GenerationDidNotAdvance,
    SessionDidNotChange,
    InvalidContract(ContractError),
}

impl fmt::Display for StreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSession => formatter.write_str("result belongs to a different session"),
            Self::WrongGeneration { expected, actual } => {
                write!(formatter, "expected generation {expected}, got {actual}")
            }
            Self::DuplicateOrOutOfOrder { last, actual } => {
                write!(formatter, "sequence {actual} is not newer than {last}")
            }
            Self::GenerationDidNotAdvance => {
                formatter.write_str("new generation must be greater than the active generation")
            }
            Self::SessionDidNotChange => {
                formatter.write_str("replacement session ID must differ from the active session")
            }
            Self::InvalidContract(error) => write!(formatter, "invalid result contract: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamError {}

/// Explicit consumer state gate. New sessions and generations must be installed deliberately;
/// packet arrival order can never switch the active clock/calibration domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultStreamGuard {
    session_id: SessionId,
    generation: u32,
    last_sequence: Option<u64>,
}

impl ResultStreamGuard {
    #[must_use]
    pub const fn new(session_id: SessionId, generation: u32) -> Self {
        Self {
            session_id,
            generation,
            last_sequence: None,
        }
    }

    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    #[must_use]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    /// Install a newer generation after receiving its session descriptor.
    ///
    /// # Errors
    ///
    /// Returns `GenerationDidNotAdvance` unless `generation` is strictly newer.
    pub fn advance_generation(&mut self, generation: u32) -> Result<(), StreamError> {
        if generation <= self.generation {
            return Err(StreamError::GenerationDidNotAdvance);
        }
        self.generation = generation;
        self.last_sequence = None;
        Ok(())
    }

    /// Install a deliberately negotiated replacement session.
    ///
    /// # Errors
    ///
    /// Returns `SessionDidNotChange` when asked to reset sequencing under the same session ID.
    pub fn replace_session(
        &mut self,
        session_id: SessionId,
        generation: u32,
    ) -> Result<(), StreamError> {
        if session_id == self.session_id {
            return Err(StreamError::SessionDidNotChange);
        }
        self.session_id = session_id;
        self.generation = generation;
        self.last_sequence = None;
        Ok(())
    }

    /// Accept a frame only if its session, generation, and sequence are current.
    ///
    /// # Errors
    ///
    /// Rejects invalid-contract, wrong-session, wrong-generation, duplicate, and out-of-order
    /// frames without advancing the accepted sequence.
    pub fn accept(&mut self, result: &NanaTrackingResult) -> Result<AcceptedFrame, StreamError> {
        result.validate().map_err(StreamError::InvalidContract)?;
        if result.session_id != self.session_id {
            return Err(StreamError::WrongSession);
        }
        if result.generation != self.generation {
            return Err(StreamError::WrongGeneration {
                expected: self.generation,
                actual: result.generation,
            });
        }
        let missing_sequences = if let Some(last) = self.last_sequence {
            if result.sequence <= last {
                return Err(StreamError::DuplicateOrOutOfOrder {
                    last,
                    actual: result.sequence,
                });
            }
            result.sequence.saturating_sub(last).saturating_sub(1)
        } else {
            0
        };
        self.last_sequence = Some(result.sequence);
        Ok(AcceptedFrame { missing_sequences })
    }
}
