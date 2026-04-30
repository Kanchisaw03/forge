use phf::{phf_map, Map};

use crate::span::Spanned;

/// Canonical token type used by the Forge lexer.
pub type Token = Spanned<TokenKind>;

/// Integer literal suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntSuffix {
    U32,
    I32,
    U64,
    I64,
    Usize,
}

/// Float literal suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatSuffix {
    F32,
    F64,
}

/// Integer literal radix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericBase {
    Decimal,
    Hex,
    Binary,
    Octal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KeywordKind {
    Fn,
    Let,
    Mut,
    If,
    Else,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    In,
    True,
    False,
    As,
    Use,
    Mod,
    Pub,
    Struct,
    Impl,
    Trait,
    Type,
    Const,
    Static,
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Usize,
}

static KEYWORDS: Map<&'static str, KeywordKind> = phf_map! {
    "fn" => KeywordKind::Fn,
    "let" => KeywordKind::Let,
    "mut" => KeywordKind::Mut,
    "if" => KeywordKind::If,
    "else" => KeywordKind::Else,
    "for" => KeywordKind::For,
    "while" => KeywordKind::While,
    "loop" => KeywordKind::Loop,
    "break" => KeywordKind::Break,
    "continue" => KeywordKind::Continue,
    "return" => KeywordKind::Return,
    "in" => KeywordKind::In,
    "true" => KeywordKind::True,
    "false" => KeywordKind::False,
    "as" => KeywordKind::As,
    "use" => KeywordKind::Use,
    "mod" => KeywordKind::Mod,
    "pub" => KeywordKind::Pub,
    "struct" => KeywordKind::Struct,
    "impl" => KeywordKind::Impl,
    "trait" => KeywordKind::Trait,
    "type" => KeywordKind::Type,
    "const" => KeywordKind::Const,
    "static" => KeywordKind::Static,
    "f32" => KeywordKind::F32,
    "f64" => KeywordKind::F64,
    "i32" => KeywordKind::I32,
    "i64" => KeywordKind::I64,
    "u32" => KeywordKind::U32,
    "u64" => KeywordKind::U64,
    "bool" => KeywordKind::Bool,
    "usize" => KeywordKind::Usize,
};

/// Complete token kind set for Forge language lexing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Keywords.
    Fn,
    Let,
    Mut,
    If,
    Else,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    In,
    True,
    False,
    As,
    Use,
    Mod,
    Pub,
    Struct,
    Impl,
    Trait,
    Type,
    Const,
    Static,

    // Primitive types.
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Usize,

    // Identifiers.
    Ident(String),

    // Literals.
    IntLiteral {
        text: String,
        base: NumericBase,
        suffix: Option<IntSuffix>,
    },
    FloatLiteral {
        text: String,
        suffix: Option<FloatSuffix>,
    },
    StringLiteral {
        raw: String,
        value: String,
    },

    // Operators.
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,

    EqEq,
    NotEq,
    Less,
    Greater,
    LessEq,
    GreaterEq,

    AndAnd,
    OrOr,
    Bang,

    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,

    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,

    DotDot,
    DotDotEq,
    Arrow,
    FatArrow,
    Eq,

    // Delimiters.
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    ColonColon,
    Dot,
    At,
    Hash,

    // End-of-file.
    Eof,
}

impl TokenKind {
    /// Returns the source lexeme for fixed and stored tokens.
    ///
    /// Errors:
    /// - Returns `None` for token kinds with no direct single lexeme.
    pub fn lexeme(&self) -> Option<&str> {
        match self {
            TokenKind::Fn => Some("fn"),
            TokenKind::Let => Some("let"),
            TokenKind::Mut => Some("mut"),
            TokenKind::If => Some("if"),
            TokenKind::Else => Some("else"),
            TokenKind::For => Some("for"),
            TokenKind::While => Some("while"),
            TokenKind::Loop => Some("loop"),
            TokenKind::Break => Some("break"),
            TokenKind::Continue => Some("continue"),
            TokenKind::Return => Some("return"),
            TokenKind::In => Some("in"),
            TokenKind::True => Some("true"),
            TokenKind::False => Some("false"),
            TokenKind::As => Some("as"),
            TokenKind::Use => Some("use"),
            TokenKind::Mod => Some("mod"),
            TokenKind::Pub => Some("pub"),
            TokenKind::Struct => Some("struct"),
            TokenKind::Impl => Some("impl"),
            TokenKind::Trait => Some("trait"),
            TokenKind::Type => Some("type"),
            TokenKind::Const => Some("const"),
            TokenKind::Static => Some("static"),
            TokenKind::F32 => Some("f32"),
            TokenKind::F64 => Some("f64"),
            TokenKind::I32 => Some("i32"),
            TokenKind::I64 => Some("i64"),
            TokenKind::U32 => Some("u32"),
            TokenKind::U64 => Some("u64"),
            TokenKind::Bool => Some("bool"),
            TokenKind::Usize => Some("usize"),
            TokenKind::Ident(text) => Some(text),
            TokenKind::IntLiteral { text, .. } => Some(text),
            TokenKind::FloatLiteral { text, .. } => Some(text),
            TokenKind::StringLiteral { raw, .. } => Some(raw),
            TokenKind::Plus => Some("+"),
            TokenKind::Minus => Some("-"),
            TokenKind::Star => Some("*"),
            TokenKind::Slash => Some("/"),
            TokenKind::Percent => Some("%"),
            TokenKind::PlusEq => Some("+="),
            TokenKind::MinusEq => Some("-="),
            TokenKind::StarEq => Some("*="),
            TokenKind::SlashEq => Some("/="),
            TokenKind::PercentEq => Some("%="),
            TokenKind::EqEq => Some("=="),
            TokenKind::NotEq => Some("!="),
            TokenKind::Less => Some("<"),
            TokenKind::Greater => Some(">"),
            TokenKind::LessEq => Some("<="),
            TokenKind::GreaterEq => Some(">="),
            TokenKind::AndAnd => Some("&&"),
            TokenKind::OrOr => Some("||"),
            TokenKind::Bang => Some("!"),
            TokenKind::Amp => Some("&"),
            TokenKind::Pipe => Some("|"),
            TokenKind::Caret => Some("^"),
            TokenKind::Tilde => Some("~"),
            TokenKind::Shl => Some("<<"),
            TokenKind::Shr => Some(">>"),
            TokenKind::AmpEq => Some("&="),
            TokenKind::PipeEq => Some("|="),
            TokenKind::CaretEq => Some("^="),
            TokenKind::ShlEq => Some("<<="),
            TokenKind::ShrEq => Some(">>="),
            TokenKind::DotDot => Some(".."),
            TokenKind::DotDotEq => Some("..="),
            TokenKind::Arrow => Some("->"),
            TokenKind::FatArrow => Some("=>"),
            TokenKind::Eq => Some("="),
            TokenKind::LParen => Some("("),
            TokenKind::RParen => Some(")"),
            TokenKind::LBracket => Some("["),
            TokenKind::RBracket => Some("]"),
            TokenKind::LBrace => Some("{"),
            TokenKind::RBrace => Some("}"),
            TokenKind::Comma => Some(","),
            TokenKind::Semicolon => Some(";"),
            TokenKind::Colon => Some(":"),
            TokenKind::ColonColon => Some("::"),
            TokenKind::Dot => Some("."),
            TokenKind::At => Some("@"),
            TokenKind::Hash => Some("#"),
            TokenKind::Eof => Some(""),
        }
    }
}

/// Returns a keyword token for `ident`, or `None` if it is a regular identifier.
///
/// This uses a compile-time perfect hash map to avoid runtime hash table
/// construction and to keep lookup branch-predictable.
pub fn keyword_from_ident(ident: &str) -> Option<TokenKind> {
    let kind = KEYWORDS.get(ident)?;
    Some(match kind {
        KeywordKind::Fn => TokenKind::Fn,
        KeywordKind::Let => TokenKind::Let,
        KeywordKind::Mut => TokenKind::Mut,
        KeywordKind::If => TokenKind::If,
        KeywordKind::Else => TokenKind::Else,
        KeywordKind::For => TokenKind::For,
        KeywordKind::While => TokenKind::While,
        KeywordKind::Loop => TokenKind::Loop,
        KeywordKind::Break => TokenKind::Break,
        KeywordKind::Continue => TokenKind::Continue,
        KeywordKind::Return => TokenKind::Return,
        KeywordKind::In => TokenKind::In,
        KeywordKind::True => TokenKind::True,
        KeywordKind::False => TokenKind::False,
        KeywordKind::As => TokenKind::As,
        KeywordKind::Use => TokenKind::Use,
        KeywordKind::Mod => TokenKind::Mod,
        KeywordKind::Pub => TokenKind::Pub,
        KeywordKind::Struct => TokenKind::Struct,
        KeywordKind::Impl => TokenKind::Impl,
        KeywordKind::Trait => TokenKind::Trait,
        KeywordKind::Type => TokenKind::Type,
        KeywordKind::Const => TokenKind::Const,
        KeywordKind::Static => TokenKind::Static,
        KeywordKind::F32 => TokenKind::F32,
        KeywordKind::F64 => TokenKind::F64,
        KeywordKind::I32 => TokenKind::I32,
        KeywordKind::I64 => TokenKind::I64,
        KeywordKind::U32 => TokenKind::U32,
        KeywordKind::U64 => TokenKind::U64,
        KeywordKind::Bool => TokenKind::Bool,
        KeywordKind::Usize => TokenKind::Usize,
    })
}
