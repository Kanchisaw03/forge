use std::fmt;

/// Byte span into the original source buffer.
///
/// Invariants:
/// - `start <= end`.
/// - `start` and `end` are byte offsets into the same source string.
/// - `line` and `col` are the 1-based line/column at `start`.
///
/// Forge stores only the start line/column because end coordinates can be
/// recovered when needed from `start..end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SourceSpan {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
    /// 1-based line number at `start`.
    pub line: u32,
    /// 1-based column number at `start`.
    pub col: u32,
}

impl SourceSpan {
    /// Creates a new span.
    ///
    /// Invariants:
    /// - Callers must provide offsets from the same source string.
    /// - `start` must be less than or equal to `end`.
    pub const fn new(start: u32, end: u32, line: u32, col: u32) -> Self {
        Self {
            start,
            end,
            line,
            col,
        }
    }

    /// Returns the span length in bytes.
    pub const fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Returns whether this span is empty.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Slices `src` using this span.
    ///
    /// Errors:
    /// - Returns `None` if this span is out of bounds for `src`.
    /// - Returns `None` if offsets are not UTF-8 char boundaries.
    pub fn try_slice<'src>(self, src: &'src str) -> Option<&'src str> {
        let start = self.start as usize;
        let end = self.end as usize;
        src.get(start..end)
    }

    /// Slices `src` using this span.
    ///
    /// Invariants:
    /// - The span must have been produced by lexing the same `src`.
    ///
    /// Errors:
    /// - Panics if `self` does not index `src` on valid UTF-8 boundaries.
    pub fn slice<'src>(self, src: &'src str) -> &'src str {
        self.try_slice(src)
            .expect("span must refer to valid source slice")
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bytes {}..{} at {}:{}",
            self.start, self.end, self.line, self.col
        )
    }
}

/// Generic node + source location wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// Payload for this syntactic object.
    pub node: T,
    /// Source location for `node`.
    pub span: SourceSpan,
}

impl<T> Spanned<T> {
    /// Constructs a spanned node.
    ///
    /// Invariants:
    /// - `span` must point to the source range that produced `node`.
    pub const fn new(node: T, span: SourceSpan) -> Self {
        Self { node, span }
    }

    /// Maps the payload while preserving the span.
    ///
    /// Errors:
    /// - This function does not fail.
    pub fn map<U, F>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }
}
