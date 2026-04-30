use thiserror::Error;

use crate::span::SourceSpan;

/// Lexer errors produced while tokenizing a Forge source file.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LexError {
    /// Encountered a character with no valid token interpretation.
    #[error("unexpected character `{ch}` at {span}")]
    UnexpectedChar { ch: char, span: SourceSpan },

    /// Reached end-of-input before a string literal was terminated.
    #[error("unterminated string literal at {span}")]
    UnterminatedString { span: SourceSpan },

    /// Encountered an unsupported escape sequence in a string literal.
    #[error("invalid escape sequence `\\{ch}` at {span}")]
    InvalidEscape { ch: char, span: SourceSpan },

    /// Numeric literal text is invalid for its format.
    #[error("invalid numeric literal `{text}` at {span}")]
    InvalidNumericLiteral { text: String, span: SourceSpan },

    /// Decimal literals with leading zeros are rejected.
    #[error("leading zero is not allowed in decimal literal at {span}")]
    LeadingZero { span: SourceSpan },
}

impl LexError {
    /// Returns the source span associated with this error.
    ///
    /// Errors:
    /// - This function does not fail.
    pub const fn span(&self) -> SourceSpan {
        match *self {
            LexError::UnexpectedChar { span, .. }
            | LexError::UnterminatedString { span }
            | LexError::InvalidEscape { span, .. }
            | LexError::InvalidNumericLiteral { span, .. }
            | LexError::LeadingZero { span } => span,
        }
    }
}

/// Parser errors used by later compilation stages.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseError {
    /// The parser found a token that does not satisfy the grammar expectation.
    #[error("unexpected token: expected {expected}, found {found} at {span}")]
    UnexpectedToken {
        expected: String,
        found: String,
        span: SourceSpan,
    },

    /// The parser reached the end of input while still expecting more syntax.
    #[error("unexpected end of input: expected {expected} at {span}")]
    UnexpectedEof { expected: String, span: SourceSpan },

    /// Catch-all parser diagnostic with a precise source span.
    #[error("parse error: {message} at {span}")]
    Message { message: String, span: SourceSpan },
}

impl ParseError {
    /// Returns the source span associated with this error.
    ///
    /// Errors:
    /// - This function does not fail.
    pub const fn span(&self) -> SourceSpan {
        match *self {
            ParseError::UnexpectedToken { span, .. }
            | ParseError::UnexpectedEof { span, .. }
            | ParseError::Message { span, .. } => span,
        }
    }
}
