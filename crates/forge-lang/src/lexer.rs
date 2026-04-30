use crate::error::LexError;
use crate::span::{SourceSpan, Spanned};
use crate::token::{
    keyword_from_ident, FloatSuffix, IntSuffix, NumericBase, Token, TokenKind,
};

#[derive(Debug, Clone, Copy)]
struct Cursor {
    pos: usize,
    line: u32,
    col: u32,
}

/// Hand-written DFA lexer for Forge source code.
///
/// The lexer is intentionally stateful and allocation-light so it can recover
/// from multiple errors in one pass while preserving exact byte spans.
pub struct Lexer<'src> {
    /// Original source text.
    pub src: &'src str,
    /// Current byte position.
    pub pos: usize,
    /// 1-based current line number.
    pub line: u32,
    /// 1-based current column number.
    pub col: u32,
    /// Collected lexical diagnostics.
    pub errors: Vec<LexError>,
}

impl<'src> Lexer<'src> {
    /// Creates a lexer positioned at the start of `src`.
    ///
    /// Invariants:
    /// - `pos` always points at a valid UTF-8 boundary in `src`.
    ///
    /// Errors:
    /// - This function does not fail.
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            pos: 0,
            line: 1,
            col: 1,
            errors: Vec::new(),
        }
    }

    /// Lexes the full source and returns all tokens plus all recovered errors.
    ///
    /// Error recovery behavior:
    /// - Unexpected characters are skipped and recorded.
    /// - Malformed literals emit diagnostics and still produce a token when
    ///   possible so downstream parsing can continue.
    pub fn lex_all(mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens = Vec::new();

        while !self.is_eof() {
            self.skip_whitespace_and_comments();
            if self.is_eof() {
                break;
            }

            if let Some(tok) = self.lex_one() {
                tokens.push(tok);
            }
        }

        let eof_span = SourceSpan::new(self.pos as u32, self.pos as u32, self.line, self.col);
        tokens.push(Spanned::new(TokenKind::Eof, eof_span));

        (tokens, self.errors)
    }

    fn lex_one(&mut self) -> Option<Token> {
        let start = self.cursor();
        let ch = self.peek_char()?;

        let kind = if is_ident_start(ch) {
            self.lex_identifier_or_keyword(start)
        } else if ch.is_ascii_digit() {
            self.lex_number(start)
        } else {
            match ch {
                '"' => self.lex_string(start)?,
                '+' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::PlusEq
                    } else {
                        TokenKind::Plus
                    }
                }
                '-' => {
                    self.advance();
                    if self.eat_if('>') {
                        TokenKind::Arrow
                    } else if self.eat_if('=') {
                        TokenKind::MinusEq
                    } else {
                        TokenKind::Minus
                    }
                }
                '*' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::StarEq
                    } else {
                        TokenKind::Star
                    }
                }
                '/' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::SlashEq
                    } else {
                        TokenKind::Slash
                    }
                }
                '%' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::PercentEq
                    } else {
                        TokenKind::Percent
                    }
                }
                '=' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::EqEq
                    } else if self.eat_if('>') {
                        TokenKind::FatArrow
                    } else {
                        TokenKind::Eq
                    }
                }
                '!' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::NotEq
                    } else {
                        TokenKind::Bang
                    }
                }
                '<' => {
                    self.advance();
                    if self.eat_if('<') {
                        if self.eat_if('=') {
                            TokenKind::ShlEq
                        } else {
                            TokenKind::Shl
                        }
                    } else if self.eat_if('=') {
                        TokenKind::LessEq
                    } else {
                        TokenKind::Less
                    }
                }
                '>' => {
                    self.advance();
                    if self.eat_if('>') {
                        if self.eat_if('=') {
                            TokenKind::ShrEq
                        } else {
                            TokenKind::Shr
                        }
                    } else if self.eat_if('=') {
                        TokenKind::GreaterEq
                    } else {
                        TokenKind::Greater
                    }
                }
                '&' => {
                    self.advance();
                    if self.eat_if('&') {
                        TokenKind::AndAnd
                    } else if self.eat_if('=') {
                        TokenKind::AmpEq
                    } else {
                        TokenKind::Amp
                    }
                }
                '|' => {
                    self.advance();
                    if self.eat_if('|') {
                        TokenKind::OrOr
                    } else if self.eat_if('=') {
                        TokenKind::PipeEq
                    } else {
                        TokenKind::Pipe
                    }
                }
                '^' => {
                    self.advance();
                    if self.eat_if('=') {
                        TokenKind::CaretEq
                    } else {
                        TokenKind::Caret
                    }
                }
                '~' => {
                    self.advance();
                    TokenKind::Tilde
                }
                '.' => {
                    self.advance();
                    if self.eat_if('.') {
                        if self.eat_if('=') {
                            TokenKind::DotDotEq
                        } else {
                            TokenKind::DotDot
                        }
                    } else {
                        TokenKind::Dot
                    }
                }
                ':' => {
                    self.advance();
                    if self.eat_if(':') {
                        TokenKind::ColonColon
                    } else {
                        TokenKind::Colon
                    }
                }
                '(' => {
                    self.advance();
                    TokenKind::LParen
                }
                ')' => {
                    self.advance();
                    TokenKind::RParen
                }
                '[' => {
                    self.advance();
                    TokenKind::LBracket
                }
                ']' => {
                    self.advance();
                    TokenKind::RBracket
                }
                '{' => {
                    self.advance();
                    TokenKind::LBrace
                }
                '}' => {
                    self.advance();
                    TokenKind::RBrace
                }
                ',' => {
                    self.advance();
                    TokenKind::Comma
                }
                ';' => {
                    self.advance();
                    TokenKind::Semicolon
                }
                '@' => {
                    self.advance();
                    TokenKind::At
                }
                '#' => {
                    self.advance();
                    TokenKind::Hash
                }
                _ => {
                    self.advance();
                    let span = self.span_from(start);
                    self.errors
                        .push(LexError::UnexpectedChar { ch, span });
                    return None;
                }
            }
        };

        Some(Spanned::new(kind, self.span_from(start)))
    }

    fn lex_identifier_or_keyword(&mut self, start: Cursor) -> TokenKind {
        self.advance();
        while let Some(ch) = self.peek_char() {
            if is_ident_continue(ch) {
                self.advance();
            } else {
                break;
            }
        }

        let text = self.slice_from(start);
        keyword_from_ident(text).unwrap_or_else(|| TokenKind::Ident(text.to_owned()))
    }

    fn lex_number(&mut self, start: Cursor) -> TokenKind {
        if self.peek_byte() == Some(b'0') {
            match self.peek_nth_byte(1) {
                Some(b'x') | Some(b'X') => {
                    self.advance_n(2);
                    return self.lex_prefixed_integer(start, NumericBase::Hex);
                }
                Some(b'b') | Some(b'B') => {
                    self.advance_n(2);
                    return self.lex_prefixed_integer(start, NumericBase::Binary);
                }
                Some(b'o') | Some(b'O') => {
                    self.advance_n(2);
                    return self.lex_prefixed_integer(start, NumericBase::Octal);
                }
                Some(next) if next.is_ascii_digit() => {
                    // Continue lexing as decimal so parser can still proceed, but
                    // report the ambiguity up front.
                    self.consume_decimal_digits();
                    let span = self.span_from(start);
                    self.errors.push(LexError::LeadingZero { span });
                    let suffix = self.consume_int_suffix();
                    if self.peek_char().is_some_and(is_ident_start) {
                        self.consume_ident_tail();
                        let span = self.span_from(start);
                        let text = self.slice_from(start).to_owned();
                        self.errors
                            .push(LexError::InvalidNumericLiteral { text, span });
                    }
                    return TokenKind::IntLiteral {
                        text: self.slice_from(start).to_owned(),
                        base: NumericBase::Decimal,
                        suffix,
                    };
                }
                _ => {}
            }
        }

        self.consume_decimal_digits();
        let mut is_float = false;

        if self.peek_byte() == Some(b'.') && self.peek_nth_byte(1) != Some(b'.') {
            if self.peek_nth_byte(1).is_some_and(|b| b.is_ascii_digit()) {
                is_float = true;
                self.advance();
                self.consume_decimal_digits();
            }
        }

        if matches!(self.peek_byte(), Some(b'e' | b'E')) {
            is_float = true;
            self.advance();
            if matches!(self.peek_byte(), Some(b'+' | b'-')) {
                self.advance();
            }

            let before_digits = self.pos;
            self.consume_decimal_digits();
            if before_digits == self.pos {
                self.consume_ident_tail();
                let span = self.span_from(start);
                let text = self.slice_from(start).to_owned();
                self.errors
                    .push(LexError::InvalidNumericLiteral { text, span });
                return TokenKind::FloatLiteral {
                    text: self.slice_from(start).to_owned(),
                    suffix: None,
                };
            }
        }

        if is_float {
            let suffix = self.consume_float_suffix();
            if suffix.is_none() {
                let _ = self.consume_int_suffix().map(|_| {
                    let span = self.span_from(start);
                    let text = self.slice_from(start).to_owned();
                    self.errors
                        .push(LexError::InvalidNumericLiteral { text, span });
                });
            }

            if self.peek_char().is_some_and(is_ident_start) {
                self.consume_ident_tail();
                let span = self.span_from(start);
                let text = self.slice_from(start).to_owned();
                self.errors
                    .push(LexError::InvalidNumericLiteral { text, span });
            }

            TokenKind::FloatLiteral {
                text: self.slice_from(start).to_owned(),
                suffix,
            }
        } else {
            let suffix = self.consume_int_suffix();
            if self.peek_char().is_some_and(is_ident_start) {
                self.consume_ident_tail();
                let span = self.span_from(start);
                let text = self.slice_from(start).to_owned();
                self.errors
                    .push(LexError::InvalidNumericLiteral { text, span });
            }

            TokenKind::IntLiteral {
                text: self.slice_from(start).to_owned(),
                base: NumericBase::Decimal,
                suffix,
            }
        }
    }

    fn lex_prefixed_integer(&mut self, start: Cursor, base: NumericBase) -> TokenKind {
        let digit_count = self.consume_base_digits(base);
        let suffix = self.consume_int_suffix();

        let has_invalid_tail = self.peek_char().is_some_and(is_ident_continue);
        if has_invalid_tail {
            self.consume_ident_tail();
        }

        if digit_count == 0 || has_invalid_tail {
            let span = self.span_from(start);
            let text = self.slice_from(start).to_owned();
            self.errors
                .push(LexError::InvalidNumericLiteral { text, span });
        }

        TokenKind::IntLiteral {
            text: self.slice_from(start).to_owned(),
            base,
            suffix,
        }
    }

    fn lex_string(&mut self, start: Cursor) -> Option<TokenKind> {
        self.advance(); // opening quote
        let mut value = String::new();

        loop {
            let Some(ch) = self.peek_char() else {
                let span = self.span_from(start);
                self.errors.push(LexError::UnterminatedString { span });
                return None;
            };

            match ch {
                '"' => {
                    self.advance();
                    let raw = self.slice_from(start).to_owned();
                    return Some(TokenKind::StringLiteral { raw, value });
                }
                '\\' => {
                    self.advance();
                    let esc_start = self.cursor();
                    let Some(esc) = self.advance() else {
                        let span = self.span_from(start);
                        self.errors.push(LexError::UnterminatedString { span });
                        return None;
                    };

                    match esc {
                        'n' => value.push('\n'),
                        't' => value.push('\t'),
                        'r' => value.push('\r'),
                        '0' => value.push('\0'),
                        '\\' => value.push('\\'),
                        '"' => value.push('"'),
                        _ => {
                            let span = self.span_from(esc_start);
                            self.errors
                                .push(LexError::InvalidEscape { ch: esc, span });
                            value.push(esc);
                        }
                    }
                }
                '\n' | '\r' => {
                    let span = self.span_from(start);
                    self.errors.push(LexError::UnterminatedString { span });
                    return None;
                }
                _ => {
                    self.advance();
                    value.push(ch);
                }
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            let mut progressed = false;

            while let Some(b) = self.peek_byte() {
                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    self.advance();
                    progressed = true;
                } else {
                    break;
                }
            }

            if self.peek_byte() == Some(b'/') && self.peek_nth_byte(1) == Some(b'/') {
                self.advance_n(2);
                while let Some(ch) = self.peek_char() {
                    if ch == '\n' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            if self.peek_byte() == Some(b'/') && self.peek_nth_byte(1) == Some(b'*') {
                self.advance_n(2);
                self.skip_block_comment();
                continue;
            }

            if !progressed {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) {
        let mut depth = 1u32;

        while let Some(ch) = self.peek_char() {
            if ch == '/' && self.peek_nth_byte(1) == Some(b'*') {
                self.advance_n(2);
                depth += 1;
                continue;
            }

            if ch == '*' && self.peek_nth_byte(1) == Some(b'/') {
                self.advance_n(2);
                depth -= 1;
                if depth == 0 {
                    return;
                }
                continue;
            }

            self.advance();
        }

        // Keep recovery simple: unmatched block comment terminator is treated as
        // end-of-file for this comment and lexing continues to EOF.
    }

    fn consume_base_digits(&mut self, base: NumericBase) -> usize {
        let mut digits = 0usize;

        while let Some(b) = self.peek_byte() {
            if b == b'_' {
                self.advance();
                continue;
            }

            let is_valid = match base {
                NumericBase::Decimal => b.is_ascii_digit(),
                NumericBase::Hex => b.is_ascii_hexdigit(),
                NumericBase::Binary => matches!(b, b'0' | b'1'),
                NumericBase::Octal => matches!(b, b'0'..=b'7'),
            };

            if !is_valid {
                break;
            }

            digits += 1;
            self.advance();
        }

        digits
    }

    fn consume_decimal_digits(&mut self) -> usize {
        self.consume_base_digits(NumericBase::Decimal)
    }

    fn consume_ident_tail(&mut self) {
        while let Some(ch) = self.peek_char() {
            if is_ident_continue(ch) {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn consume_int_suffix(&mut self) -> Option<IntSuffix> {
        let suffix = if self.starts_with_boundary("usize") {
            self.advance_n(5);
            Some(IntSuffix::Usize)
        } else if self.starts_with_boundary("u32") {
            self.advance_n(3);
            Some(IntSuffix::U32)
        } else if self.starts_with_boundary("i32") {
            self.advance_n(3);
            Some(IntSuffix::I32)
        } else if self.starts_with_boundary("u64") {
            self.advance_n(3);
            Some(IntSuffix::U64)
        } else if self.starts_with_boundary("i64") {
            self.advance_n(3);
            Some(IntSuffix::I64)
        } else {
            None
        };

        suffix
    }

    fn consume_float_suffix(&mut self) -> Option<FloatSuffix> {
        let suffix = if self.starts_with_boundary("f32") {
            self.advance_n(3);
            Some(FloatSuffix::F32)
        } else if self.starts_with_boundary("f64") {
            self.advance_n(3);
            Some(FloatSuffix::F64)
        } else {
            None
        };

        suffix
    }

    fn starts_with_boundary(&self, s: &str) -> bool {
        let rest = &self.src[self.pos..];
        if !rest.starts_with(s) {
            return false;
        }

        let next_pos = self.pos + s.len();
        let Some(next) = self.src.get(next_pos..) else {
            return true;
        };

        match next.chars().next() {
            None => true,
            Some(ch) => !is_ident_continue(ch),
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn cursor(&self) -> Cursor {
        Cursor {
            pos: self.pos,
            line: self.line,
            col: self.col,
        }
    }

    fn span_from(&self, start: Cursor) -> SourceSpan {
        SourceSpan::new(start.pos as u32, self.pos as u32, start.line, start.col)
    }

    fn slice_from(&self, start: Cursor) -> &str {
        &self.src[start.pos..self.pos]
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn peek_nth_byte(&self, n: usize) -> Option<u8> {
        self.src.as_bytes().get(self.pos + n).copied()
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn eat_if(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn advance_n(&mut self, n: usize) {
        for _ in 0..n {
            let _ = self.advance();
        }
    }

    fn advance(&mut self) -> Option<char> {
        let byte = *self.src.as_bytes().get(self.pos)?;

        // Performance rationale: source text is overwhelmingly ASCII. This
        // fast path avoids UTF-8 decoding for the hot lexing loop.
        if byte.is_ascii() {
            self.pos += 1;
            let ch = byte as char;
            if ch == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            return Some(ch);
        }

        let ch = self.src[self.pos..].chars().next()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> (Vec<Token>, Vec<LexError>) {
        Lexer::new(src).lex_all()
    }

    fn kinds(tokens: &[Token]) -> Vec<TokenKind> {
        tokens.iter().map(|t| t.node.clone()).collect()
    }

    #[test]
    fn every_keyword_tokenizes_correctly() {
        let src = "fn let mut if else for while loop break continue return in true false as use mod pub struct impl trait type const static f32 f64 i32 i64 u32 u64 bool usize";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let expected = vec![
            TokenKind::Fn,
            TokenKind::Let,
            TokenKind::Mut,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::For,
            TokenKind::While,
            TokenKind::Loop,
            TokenKind::Break,
            TokenKind::Continue,
            TokenKind::Return,
            TokenKind::In,
            TokenKind::True,
            TokenKind::False,
            TokenKind::As,
            TokenKind::Use,
            TokenKind::Mod,
            TokenKind::Pub,
            TokenKind::Struct,
            TokenKind::Impl,
            TokenKind::Trait,
            TokenKind::Type,
            TokenKind::Const,
            TokenKind::Static,
            TokenKind::F32,
            TokenKind::F64,
            TokenKind::I32,
            TokenKind::I64,
            TokenKind::U32,
            TokenKind::U64,
            TokenKind::Bool,
            TokenKind::Usize,
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn forge_contextual_words_after_at_are_identifiers() {
        let src = "@kernel @shared @global @register";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let expected = vec![
            TokenKind::At,
            TokenKind::Ident("kernel".to_owned()),
            TokenKind::At,
            TokenKind::Ident("shared".to_owned()),
            TokenKind::At,
            TokenKind::Ident("global".to_owned()),
            TokenKind::At,
            TokenKind::Ident("register".to_owned()),
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn every_operator_tokenizes_correctly() {
        let src = "+ - * / % += -= *= /= %= == != < > <= >= && || ! & | ^ ~ << >> &= |= ^= <<= >>= .. ..= -> => = ( ) [ ] { } , ; : :: . @ #";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let expected = vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::Percent,
            TokenKind::PlusEq,
            TokenKind::MinusEq,
            TokenKind::StarEq,
            TokenKind::SlashEq,
            TokenKind::PercentEq,
            TokenKind::EqEq,
            TokenKind::NotEq,
            TokenKind::Less,
            TokenKind::Greater,
            TokenKind::LessEq,
            TokenKind::GreaterEq,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Bang,
            TokenKind::Amp,
            TokenKind::Pipe,
            TokenKind::Caret,
            TokenKind::Tilde,
            TokenKind::Shl,
            TokenKind::Shr,
            TokenKind::AmpEq,
            TokenKind::PipeEq,
            TokenKind::CaretEq,
            TokenKind::ShlEq,
            TokenKind::ShrEq,
            TokenKind::DotDot,
            TokenKind::DotDotEq,
            TokenKind::Arrow,
            TokenKind::FatArrow,
            TokenKind::Eq,
            TokenKind::LParen,
            TokenKind::RParen,
            TokenKind::LBracket,
            TokenKind::RBracket,
            TokenKind::LBrace,
            TokenKind::RBrace,
            TokenKind::Comma,
            TokenKind::Semicolon,
            TokenKind::Colon,
            TokenKind::ColonColon,
            TokenKind::Dot,
            TokenKind::At,
            TokenKind::Hash,
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn integer_literals_with_and_without_suffixes() {
        let src = "42 0xFF 0b1010 0o77 42u32 7i32 9u64 11i64 13usize";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let expected = vec![
            TokenKind::IntLiteral {
                text: "42".to_owned(),
                base: NumericBase::Decimal,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "0xFF".to_owned(),
                base: NumericBase::Hex,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "0b1010".to_owned(),
                base: NumericBase::Binary,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "0o77".to_owned(),
                base: NumericBase::Octal,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "42u32".to_owned(),
                base: NumericBase::Decimal,
                suffix: Some(IntSuffix::U32),
            },
            TokenKind::IntLiteral {
                text: "7i32".to_owned(),
                base: NumericBase::Decimal,
                suffix: Some(IntSuffix::I32),
            },
            TokenKind::IntLiteral {
                text: "9u64".to_owned(),
                base: NumericBase::Decimal,
                suffix: Some(IntSuffix::U64),
            },
            TokenKind::IntLiteral {
                text: "11i64".to_owned(),
                base: NumericBase::Decimal,
                suffix: Some(IntSuffix::I64),
            },
            TokenKind::IntLiteral {
                text: "13usize".to_owned(),
                base: NumericBase::Decimal,
                suffix: Some(IntSuffix::Usize),
            },
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn malformed_prefixed_literals_report_invalid_numeric() {
        let src = "0b102 0xGG 0o78";
        let (tokens, errors) = lex(src);

        assert_eq!(errors.len(), 3, "unexpected errors: {errors:#?}");
        assert!(
            errors
                .iter()
                .all(|e| matches!(e, LexError::InvalidNumericLiteral { .. }))
        );

        let expected = vec![
            TokenKind::IntLiteral {
                text: "0b102".to_owned(),
                base: NumericBase::Binary,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "0xGG".to_owned(),
                base: NumericBase::Hex,
                suffix: None,
            },
            TokenKind::IntLiteral {
                text: "0o78".to_owned(),
                base: NumericBase::Octal,
                suffix: None,
            },
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn float_literals_with_exponent_and_suffixes() {
        let src = "3.14 3.14f32 3.14e-5 3.14e-5f32 6.0f64";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let expected = vec![
            TokenKind::FloatLiteral {
                text: "3.14".to_owned(),
                suffix: None,
            },
            TokenKind::FloatLiteral {
                text: "3.14f32".to_owned(),
                suffix: Some(FloatSuffix::F32),
            },
            TokenKind::FloatLiteral {
                text: "3.14e-5".to_owned(),
                suffix: None,
            },
            TokenKind::FloatLiteral {
                text: "3.14e-5f32".to_owned(),
                suffix: Some(FloatSuffix::F32),
            },
            TokenKind::FloatLiteral {
                text: "6.0f64".to_owned(),
                suffix: Some(FloatSuffix::F64),
            },
            TokenKind::Eof,
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn string_literals_support_all_required_escapes() {
        let src = "\"a\\n\\t\\\\\\\"b\\r\\0\"";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let first = &tokens[0].node;
        match first {
            TokenKind::StringLiteral { raw, value } => {
                assert_eq!(raw, "\"a\\n\\t\\\\\\\"b\\r\\0\"");
                assert_eq!(value, "a\n\t\\\"b\r\0");
            }
            other => panic!("expected string literal, got {other:?}"),
        }
    }

    #[test]
    fn recovers_after_multiple_errors_and_keeps_valid_tokens() {
        let src = "let x = 1; $ \"bad\\q\" 0xGG fn ok() {}";
        let (tokens, errors) = lex(src);

        assert_eq!(errors.len(), 3, "unexpected errors: {errors:#?}");
        assert!(matches!(errors[0], LexError::UnexpectedChar { .. }));
        assert!(matches!(errors[1], LexError::InvalidEscape { .. }));
        assert!(matches!(errors[2], LexError::InvalidNumericLiteral { .. }));

        let token_kinds = kinds(&tokens);
        assert!(token_kinds.contains(&TokenKind::Let));
        assert!(token_kinds.contains(&TokenKind::Fn));
        assert!(token_kinds.contains(&TokenKind::Ident("ok".to_owned())));
    }

    #[test]
    fn token_span_matches_source_text_for_all_tokens() {
        let src = "fn foo(a: &[f32], b: i32) { let x = \"hi\\n\"; x += 0x1Fu32; }";
        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        for token in &tokens {
            let sliced = token.span.slice(src);
            let lexeme = token
                .node
                .lexeme()
                .expect("every token kind in this lexer has a lexeme");
            assert_eq!(sliced, lexeme, "mismatch for token {token:?}");
        }
    }

    #[test]
    fn matmul_kernel_token_sequence_is_exact() {
        let src = r#"
@kernel
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], N: u32) {
    let row = thread::x();
    let col = thread::y();

    if row < N && col < N {
        let mut sum: f32 = 0.0;
        for k in 0u32..N {
            sum += a[row * N + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}
"#;

        let (tokens, errors) = lex(src);
        assert!(errors.is_empty(), "expected no lex errors: {errors:?}");

        let lexemes: Vec<String> = tokens.iter().map(|t| t.span.slice(src).to_owned()).collect();

        let expected = vec![
            "@", "kernel", "fn", "matmul", "(", "a", ":", "&", "[", "f32", "]", ",", "b", ":", "&", "[", "f32", "]", ",", "out", ":", "&", "mut", "[", "f32", "]", ",", "N", ":", "u32", ")", "{",
            "let", "row", "=", "thread", "::", "x", "(", ")", ";",
            "let", "col", "=", "thread", "::", "y", "(", ")", ";",
            "if", "row", "<", "N", "&&", "col", "<", "N", "{",
            "let", "mut", "sum", ":", "f32", "=", "0.0", ";",
            "for", "k", "in", "0u32", "..", "N", "{",
            "sum", "+=", "a", "[", "row", "*", "N", "+", "k", "]", "*", "b", "[", "k", "*", "N", "+", "col", "]", ";",
            "}",
            "out", "[", "row", "*", "N", "+", "col", "]", "=", "sum", ";",
            "}", "}", "",
        ];

        assert_eq!(lexemes, expected);
    }
}
