//! Forge language frontend primitives.
//!
//! This crate currently provides source spans, token definitions, and a
//! hand-written lexer with multi-error recovery.

pub mod ast;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod resolve;
pub mod span;
pub mod token;

pub use ast::*;
pub use error::{LexError, ParseError};
pub use lexer::Lexer;
pub use parser::ParseSession;
pub use resolve::{resolve_module, ResolutionError, ResolutionErrorKind, ScopeTree};
pub use span::{SourceSpan, Spanned};
pub use token::{
    keyword_from_ident, FloatSuffix, IntSuffix, NumericBase, Token, TokenKind,
};
