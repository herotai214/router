//! L1 prefix encoding cache: router-side token IDs, not the worker's KV
//! prefix cache.
//!
//! Hugging Face tokenizers (and fastokens) split the input on added tokens
//! before normalizing, pre-tokenizing and running the model, and each text
//! piece is processed on its own. So when a prefix `P` ends right after a
//! *boundary* added token, `encode(P + S) == encode(P) ++ encode(S)` with
//! `add_special_tokens = false`. For Qwen-style templates the boundaries are
//! `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<think>`, ...
//!
//! The cache is a radix tree over rendered prompt text. Edges end at
//! boundaries and keep the token IDs of their text plus *marks*: the
//! `(byte, token)` offsets of the boundaries inside the edge. A lookup
//! follows the request text through the tree and may stop at any mark, so a
//! request that edits message `k` still reuses messages `0..k`. Only the
//! remaining text after the matched boundary is encoded, in one call; the
//! token offsets of its boundaries come from the boundary token IDs in that
//! encoding, and the new boundaries up to the insert cap are added to the
//! tree as one edge.
//!
//! Children live in one tree-wide map keyed by (parent, first segment), so
//! nodes carry no per-node maps. Leaves are evicted in LRU order from a heap,
//! and `max_bytes` covers node contents plus container capacity.
//!
//! Cut positions come from an automaton over all added tokens that matches
//! like the tokenizer's added-token split. A token is a boundary only when
//! no longer added token can contain it (see [`boundary_rejections`]) and it
//! has no strip/word flags. A load-time probe compares split and unsplit
//! encodes for each candidate in synthetic contexts and drops any candidate
//! that disagrees.
//!
//! Not covered yet: multimodal placeholders (the gRPC path is text-only),
//! tiktoken/tekken tokenizers, boundaries inside messages, and HTTP workers.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Once};

use aho_corasick::{AhoCorasick, Input, MatchKind};
use anyhow::{bail, Context, Result};
use parking_lot::{Mutex, RwLock};
use regex::Regex;
use serde_json::Value;
use vllm_text::backend::hf::{ResolvedModelFiles, TokenizerSource};
use vllm_tokenizer::Tokenizer;

use crate::metrics::TokenizerMetrics;

/// L1 budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1CacheConfig {
    /// Estimated heap bytes retained across models: node contents plus the
    /// tree's containers. Evictions run after each insert until it holds.
    /// Must be at least 1.
    pub max_bytes: usize,
    /// Tree nodes retained across models. Must be at least 1.
    pub max_nodes: usize,
    /// Estimated bytes one request may add, including a new namespace root,
    /// interior boundary marks, and a split node. At most `max_bytes`.
    pub max_insert_bytes: usize,
    /// Boundary segments one request may add. Must be at least 1.
    pub max_insert_segments: usize,
    /// Child key hashing strategy. The default avoids hashing whole long
    /// segments; `Full` is useful for collision-heavy diagnostics.
    pub hash_strategy: L1HashStrategy,
    /// Re-encode every prompt in full and compare (debugging only).
    pub verify: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L1HashStrategy {
    FirstLast64,
    Full,
}

impl Default for L1CacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_nodes: 65_536,
            max_insert_bytes: 32 * 1024 * 1024,
            max_insert_segments: 4096,
            hash_strategy: L1HashStrategy::FirstLast64,
            verify: false,
        }
    }
}

impl L1CacheConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_bytes == 0 || self.max_nodes == 0 || self.max_insert_segments == 0 {
            bail!("L1 limits must be positive");
        }
        if self.max_insert_bytes == 0 || self.max_insert_bytes > self.max_bytes {
            bail!("L1 max_insert_bytes must be positive and at most max_bytes");
        }
        Ok(())
    }
}

/// L1 counters and occupancy across models.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct L1CacheStats {
    /// Prompts that had at least one boundary and went through the tree.
    pub lookups: u64,
    /// Lookups that reused at least one cached token.
    pub hits: u64,
    /// Prompt tokens taken from the tree.
    pub cached_tokens: u64,
    /// Prompt tokens produced by the tokenizer on the L1 path.
    pub encoded_tokens: u64,
    /// Nodes removed to satisfy the byte or node budget.
    pub evictions: u64,
    /// Requests whose next segment alone exceeded `max_insert_bytes`.
    pub oversized: u64,
    /// Verify-mode mismatches (each disables L1 for that model).
    pub verify_mismatches: u64,
    /// Prompts on an L1-enabled model that skipped the tree (any reason).
    pub bypasses: u64,
    pub nodes: usize,
    /// Same estimate `max_bytes` bounds.
    pub bytes: usize,
}

/// Why a prompt on an L1-enabled model did not go through the tree. An L0
/// hit is not a bypass: L1 is not consulted at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L1Bypass {
    /// The renderer produced token IDs (e.g. Harmony), not text.
    TokenIds,
    /// BOS/EOS post-processing is not prefix-stable.
    AddSpecialTokens,
    /// The prompt contains no boundary token.
    NoBoundary,
    /// L1 was turned off for this model after a verify mismatch.
    Disabled,
}

impl L1Bypass {
    fn as_str(self) -> &'static str {
        match self {
            Self::TokenIds => "token_ids",
            Self::AddSpecialTokens => "add_special_tokens",
            Self::NoBoundary => "no_boundary",
            Self::Disabled => "disabled",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AddedTokenSpec {
    pub content: String,
    pub lstrip: bool,
    pub rstrip: bool,
    pub single_word: bool,
    pub normalized: bool,
}

/// A model's added tokens and whether its normalizer leaves text unchanged.
#[derive(Debug, Clone, Default)]
pub(crate) struct AddedTokenSet {
    pub tokens: Vec<AddedTokenSpec>,
    pub identity_normalizer: bool,
}

/// Finds cut positions: byte offsets right after a boundary token.
///
/// The automaton holds every added token and matches leftmost-longest like
/// the tokenizer's added-token split, so a boundary only counts where the
/// tokenizer itself splits it out.
pub(crate) struct BoundaryMatcher {
    automaton: AhoCorasick,
    patterns: Vec<String>,
    /// Per pattern: whether a cut follows it.
    cut: Vec<bool>,
    tokens: Vec<String>,
    /// Per pattern: the token ID of a boundary, once resolved.
    cut_ids: Vec<Option<u32>>,
    /// Bitset of every resolved boundary token ID.
    cut_id_bits: Vec<u64>,
}

impl std::fmt::Debug for BoundaryMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundaryMatcher")
            .field("tokens", &self.tokens)
            .finish()
    }
}

impl BoundaryMatcher {
    pub(crate) fn new(added: &AddedTokenSet) -> Result<Self> {
        let mut tokens = Vec::new();
        for (token, rejection) in added.tokens.iter().zip(boundary_rejections(added)) {
            match rejection {
                None => tokens.push(token.content.clone()),
                Some(reason) => {
                    tracing::debug!(token = %token.content, reason, "not an L1 boundary")
                }
            }
        }
        let patterns = added
            .tokens
            .iter()
            .filter(|token| !token.content.is_empty())
            .map(|token| token.content.clone())
            .collect();
        Self::build(patterns, tokens)
    }

    fn build(mut patterns: Vec<String>, mut tokens: Vec<String>) -> Result<Self> {
        tokens.sort();
        tokens.dedup();
        if tokens.is_empty() {
            bail!("no added token is a safe encoding boundary");
        }
        patterns.sort();
        patterns.dedup();
        let cut = patterns
            .iter()
            .map(|pattern| tokens.binary_search(pattern).is_ok())
            .collect();
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)?;
        Ok(Self {
            automaton,
            patterns,
            cut,
            tokens,
            cut_ids: Vec::new(),
            cut_id_bits: Vec::new(),
        })
    }

    pub(crate) fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Record each boundary's token ID so cut offsets can be read from one
    /// encode (see [`Self::token_offsets`]).
    pub(crate) fn with_token_ids(mut self, tokenizer: &dyn Tokenizer) -> Self {
        self.cut_ids = self
            .patterns
            .iter()
            .zip(&self.cut)
            .map(|(pattern, &cut)| cut.then(|| tokenizer.token_to_id(pattern)).flatten())
            .collect();
        let mut bits = Vec::new();
        for id in self.cut_ids.iter().flatten().map(|&id| id as usize) {
            if bits.len() <= id / 64 {
                bits.resize(id / 64 + 1, 0u64);
            }
            bits[id / 64] |= 1 << (id % 64);
        }
        self.cut_id_bits = bits;
        self
    }

    /// Cut positions in `text`, which must itself start at a cut position.
    #[cfg(test)]
    pub(crate) fn boundaries(&self, text: &str) -> Vec<usize> {
        self.cuts(text, 0).into_iter().map(|(end, _)| end).collect()
    }

    /// Cuts in `text[from..]` as (end, pattern). `from` must be a cut
    /// position: the tokenizer's leftmost-longest split restarts there, so
    /// the matches after it do not depend on the text before it.
    fn cuts(&self, text: &str, from: usize) -> Vec<(usize, usize)> {
        self.automaton
            .find_iter(Input::new(text).range(from..))
            .filter(|m| self.cut[m.pattern().as_usize()])
            .map(|m| (m.end(), m.pattern().as_usize()))
            .collect()
    }

    /// The first cut after `from`, which must be a cut position.
    fn first_cut(&self, text: &str, from: usize) -> Option<usize> {
        self.automaton
            .find_iter(Input::new(text).range(from..))
            .find(|m| self.cut[m.pattern().as_usize()])
            .map(|m| m.end())
    }

    /// Token offsets right after each of `cuts` in `ids`, the encoding of
    /// text that starts at a cut. The tokenizer splits boundary tokens out
    /// before its model runs, so each cut ends with that token's ID. `None`
    /// if the IDs do not line up one to one with `cuts`, e.g. when the model
    /// can also produce a boundary ID from plain text.
    fn token_offsets(&self, cuts: &[(usize, usize)], ids: &[u32]) -> Option<Vec<usize>> {
        if self.cut_id_bits.is_empty() {
            return None;
        }
        let mut offsets = Vec::with_capacity(cuts.len());
        for (i, &id) in ids.iter().enumerate() {
            let id_bit = self.cut_id_bits.get(id as usize / 64).copied().unwrap_or(0);
            if id_bit >> (id % 64) & 1 == 0 {
                continue;
            }
            let &(_, pattern) = cuts.get(offsets.len())?;
            if self.cut_ids[pattern] != Some(id) {
                return None;
            }
            offsets.push(i + 1);
        }
        (offsets.len() == cuts.len()).then_some(offsets)
    }

    /// Keep only boundaries for which splitting gives the same IDs as a full
    /// encode of synthetic prompts.
    pub(crate) fn probed(self, tokenizer: &dyn Tokenizer) -> Result<Self> {
        const BEFORE: [&str; 5] = ["", "hi", "Hello, world. ", "x\n", "数字 123"];
        const AFTER: [&str; 11] = [
            "",
            "hi",
            " world",
            "\n\nnext",
            "\u{301}e",
            "\u{338}",
            "123 456",
            " \t ",
            "日本語",
            "😀!",
            "'s it",
        ];
        let mut kept = Vec::with_capacity(self.tokens.len());
        'tokens: for token in &self.tokens {
            for before in BEFORE {
                let prefix = format!("{before}{token}");
                let head = tokenizer.encode(&prefix, false)?;
                for after in AFTER {
                    let mut split = head.clone();
                    split.extend(tokenizer.encode(after, false)?);
                    if tokenizer.encode(&format!("{prefix}{after}"), false)? != split {
                        tracing::warn!(token, after, "L1 boundary failed probe; skipping");
                        continue 'tokens;
                    }
                }
            }
            kept.push(token.clone());
        }
        Ok(Self::build(self.patterns, kept)?.with_token_ids(tokenizer))
    }
}

/// Why each added token cannot be a boundary (`None` if it can).
///
/// fastokens matches all added tokens in one leftmost-longest pass; Hugging
/// Face matches non-normalized tokens first and normalized ones after
/// normalization. When no token of one group can overlap a token of the
/// other, both split exactly like [`BoundaryMatcher`]. Then a cut after `t`
/// only fails if a longer added token contains `t` before its end: it can
/// win in `P + S` but does not fit in `P`. Otherwise no added token may
/// overlap `t` at all.
pub(crate) fn boundary_rejections(added: &AddedTokenSet) -> Vec<Option<&'static str>> {
    let contents: Vec<&str> = added
        .tokens
        .iter()
        .map(|token| token.content.as_str())
        .filter(|content| !content.is_empty())
        .collect();
    let (normalized, raw): (Vec<&AddedTokenSpec>, Vec<&AddedTokenSpec>) =
        added.tokens.iter().partition(|token| token.normalized);
    let separate = normalized
        .iter()
        .all(|y| raw.iter().all(|x| !overlaps(&x.content, &y.content)));
    let covered = covered_tokens(&contents);
    let any_single_word = added.tokens.iter().any(|token| token.single_word);
    let rejection = |token: &AddedTokenSpec| {
        let t = token.content.as_str();
        if t.is_empty() {
            return Some("empty");
        }
        if token.rstrip {
            return Some("rstrip absorbs whitespace after the cut");
        }
        if token.single_word {
            return Some("single_word depends on the next character");
        }
        if token.lstrip {
            return Some("lstrip (excluded conservatively)");
        }
        if token.normalized && !added.identity_normalizer {
            return Some("normalized and the normalizer changes text");
        }
        if token.normalized && !separate {
            return Some("normalized tokens overlap non-normalized ones");
        }
        // Strip flags on neighbouring tokens absorb adjacent whitespace.
        if t.starts_with(char::is_whitespace) || t.ends_with(char::is_whitespace) {
            return Some("starts or ends with whitespace");
        }
        // A single_word token right after the cut sees `t`'s last character
        // in `P + S` but the start of input in `S`.
        if any_single_word && ends_with_word(t) {
            return Some("a single_word token could follow");
        }
        if covered.contains(t) {
            return Some("a longer added token contains it");
        }
        if !separate
            && (char_offsets(t).any(|k| t.starts_with(&t[k..]))
                || contents.iter().any(|&x| x != t && reaches_into(x, t)))
        {
            return Some("another added token can overlap it");
        }
        None
    };
    added.tokens.iter().map(rejection).collect()
}

/// Hugging Face's `single_word` test: Unicode `\w`, which also covers
/// combining marks, connector punctuation and joiners.
fn ends_with_word(text: &str) -> bool {
    static ENDS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w$").unwrap());
    ENDS_WITH_WORD.is_match(text)
}

/// Tokens that occur inside another added token, ending before its end.
fn covered_tokens<'a>(contents: &[&'a str]) -> HashSet<&'a str> {
    let Ok(automaton) = AhoCorasick::new(contents) else {
        return contents.iter().copied().collect();
    };
    let mut covered = HashSet::new();
    for x in contents {
        for m in automaton.find_overlapping_iter(x) {
            if m.end() < x.len() {
                covered.insert(contents[m.pattern().as_usize()]);
            }
        }
    }
    covered
}

/// Whether occurrences of `a` and `b` can overlap in some text.
fn overlaps(a: &str, b: &str) -> bool {
    a == b || reaches_into(a, b) || reaches_into(b, a)
}

/// Whether a match of `x` can start before an occurrence of `t` and reach
/// into it, or start at the same byte and be longer.
fn reaches_into(x: &str, t: &str) -> bool {
    if x.len() > t.len() && x.starts_with(t) {
        return true;
    }
    char_offsets(x).any(|k| {
        let suffix = &x[k..];
        t.starts_with(suffix) || suffix.starts_with(t)
    })
}

fn char_offsets(s: &str) -> impl Iterator<Item = usize> + '_ {
    s.char_indices().map(|(i, _)| i).skip(1)
}

/// Build a boundary matcher from the model's tokenizer files, or explain why
/// L1 cannot be used for it.
pub(crate) async fn load_boundary_matcher(model: &str) -> Result<BoundaryMatcher> {
    BoundaryMatcher::new(&load_added_tokens(model).await?)
}

/// Added tokens as vllm-tokenizer sees them, after config checks.
pub(crate) async fn load_added_tokens(model: &str) -> Result<AddedTokenSet> {
    let files = ResolvedModelFiles::new(model).await?;
    let TokenizerSource::HuggingFace(path) = files.tokenizer else {
        bail!("L1 supports tokenizer.json models only");
    };
    let json: Value = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    check_tokenizer_json(&json)?;
    let mut added = added_tokens(&json["added_tokens"], None);
    // vllm-tokenizer merges extra tokens from the sibling tokenizer_config.json.
    let config_path = path
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer_config.json");
    if let Ok(raw) = std::fs::read(&config_path) {
        if let Ok(config) = serde_json::from_slice::<Value>(&raw) {
            let known: Vec<u64> = json["added_tokens"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|token| token["id"].as_u64())
                .collect();
            added.extend(added_tokens(&config["added_tokens_decoder"], Some(&known)));
        }
    }
    Ok(AddedTokenSet {
        tokens: added,
        identity_normalizer: is_identity_normalizer(&json["normalizer"]),
    })
}

fn is_identity_normalizer(normalizer: &Value) -> bool {
    match normalizer["type"].as_str() {
        None => normalizer.is_null(),
        Some("Sequence") => normalizer["normalizers"]
            .as_array()
            .is_some_and(|steps| steps.iter().all(is_identity_normalizer)),
        _ => false,
    }
}

fn check_tokenizer_json(json: &Value) -> Result<()> {
    if !super::tokenizer_determinism::deterministic_hf_model(&json["model"]) {
        bail!("tokenizer model is not known to be deterministic");
    }
    if !json["truncation"].is_null() || !json["padding"].is_null() {
        bail!("tokenizer.json enables truncation or padding");
    }
    if uses_first_only_metaspace(&json["pre_tokenizer"]) {
        bail!("Metaspace prepend_scheme=first depends on the split position");
    }
    Ok(())
}

fn uses_first_only_metaspace(pre_tokenizer: &Value) -> bool {
    match pre_tokenizer["type"].as_str() {
        Some("Metaspace") => pre_tokenizer["prepend_scheme"].as_str() == Some("first"),
        Some("Sequence") => pre_tokenizer["pretokenizers"]
            .as_array()
            .into_iter()
            .flatten()
            .any(uses_first_only_metaspace),
        _ => false,
    }
}

/// Parse `added_tokens` (array) or `added_tokens_decoder` (id-keyed map),
/// skipping IDs in `skip`.
fn added_tokens(value: &Value, skip: Option<&[u64]>) -> Vec<AddedTokenSpec> {
    let parse = |token: &Value| {
        Some(AddedTokenSpec {
            content: token["content"].as_str()?.to_string(),
            lstrip: token["lstrip"].as_bool().unwrap_or(false),
            rstrip: token["rstrip"].as_bool().unwrap_or(false),
            single_word: token["single_word"].as_bool().unwrap_or(false),
            // vllm-tokenizer fills a missing field with false before either
            // backend loads. Hugging Face's own default is `!special`.
            normalized: token["normalized"].as_bool().unwrap_or(false),
        })
    };
    match value {
        Value::Array(tokens) => tokens.iter().filter_map(parse).collect(),
        Value::Object(tokens) => tokens
            .iter()
            .filter(|(id, _)| {
                // vllm-tokenizer parses decoder keys as `u32` and drops the
                // entry when that fails. A wider parse would admit a boundary
                // the tokenizer never installs.
                let Ok(id) = id.parse::<u32>() else {
                    return false;
                };
                !skip.unwrap_or_default().contains(&(u64::from(id)))
            })
            .filter_map(|(_, token)| parse(token))
            .collect(),
        _ => Vec::new(),
    }
}

const NO_NODE: usize = usize::MAX;

/// Bytes the allocator hands out for an `n`-byte request: 16-byte chunks
/// with an 8-byte header and a 32-byte minimum, as in glibc malloc.
const fn heap_bytes(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let chunk = (n + 8 + 15) & !15;
    if chunk < 32 {
        32
    } else {
        chunk
    }
}

/// Heap bytes a node owns: the boxed node, its edge text, its ID slice with
/// the `Arc` counts, its marks, and a namespace salt.
fn node_bytes(text: usize, tokens: usize, marks: usize, salt: Option<usize>) -> usize {
    heap_bytes(size_of::<Node>())
        + heap_bytes(text)
        + heap_bytes(2 * size_of::<usize>() + tokens * size_of::<u32>())
        + heap_bytes(marks * size_of::<(u32, u32)>())
        + salt.map_or(0, |len| heap_bytes(2 * size_of::<usize>() + len))
}

/// Length of the common prefix of `a` and `b`.
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    const CHUNK: usize = 64;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + CHUNK <= n && a[i..i + CHUNK] == b[i..i + CHUNK] {
        i += CHUNK;
    }
    i + a[i..n]
        .iter()
        .zip(&b[i..n])
        .take_while(|(x, y)| x == y)
        .count()
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Namespace {
    instance: u64,
    /// Shared by the node and the roots map, so the bytes are counted once.
    salt: Option<Arc<str>>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct SegmentKey {
    len: usize,
    fingerprint: u64,
}

/// Seeded per process, so prompts cannot be crafted to collide.
static SEGMENT_HASH: LazyLock<foldhash::quality::RandomState> = LazyLock::new(Default::default);

fn segment_key(segment: &str, strategy: L1HashStrategy) -> SegmentKey {
    let bytes = segment.as_bytes();
    let mut hasher = SEGMENT_HASH.build_hasher();
    match strategy {
        L1HashStrategy::Full => bytes.hash(&mut hasher),
        L1HashStrategy::FirstLast64 if bytes.len() <= 128 => bytes.hash(&mut hasher),
        L1HashStrategy::FirstLast64 => {
            bytes[..64].hash(&mut hasher);
            bytes[bytes.len() - 64..].hash(&mut hasher);
        }
    }
    SegmentKey {
        len: bytes.len(),
        fingerprint: hasher.finish(),
    }
}

/// A parent and the key of a child's first segment.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct EdgeKey {
    parent: usize,
    segment: SegmentKey,
}

struct Node {
    parent: usize,
    /// Next child of `parent` whose first segment has the same key.
    next_same_key: usize,
    child_count: usize,
    /// Edge text; it ends at a boundary (empty for namespace roots).
    text: Box<str>,
    ids: Arc<[u32]>,
    /// Boundaries strictly inside the edge as (byte, token) offsets.
    marks: Box<[(u32, u32)]>,
    /// Tick of the last insert before this node was last used.
    last_used: AtomicU64,
    /// Set on namespace roots.
    namespace: Option<Namespace>,
}

impl Node {
    fn new(
        parent: usize,
        text: Box<str>,
        ids: Arc<[u32]>,
        marks: Box<[(u32, u32)]>,
        tick: u64,
        namespace: Option<Namespace>,
    ) -> Self {
        Self {
            parent,
            next_same_key: NO_NODE,
            child_count: 0,
            text,
            ids,
            marks,
            last_used: AtomicU64::new(tick),
            namespace,
        }
    }

    fn first_segment(&self) -> &str {
        match self.marks.first() {
            Some(&(byte, _)) => &self.text[..byte as usize],
            None => &self.text,
        }
    }

    fn is_leaf(&self) -> bool {
        self.namespace.is_none() && self.child_count == 0
    }

    fn bytes(&self) -> usize {
        let salt = self.namespace.as_ref().and_then(|ns| ns.salt.as_ref());
        node_bytes(
            self.text.len(),
            self.ids.len(),
            self.marks.len(),
            salt.map(|salt| salt.len()),
        )
    }

    /// Skips the write when the node is already current, so hot prefixes
    /// are not written by every lookup.
    fn touch(&self, tick: u64) {
        if self.last_used.load(Ordering::Relaxed) < tick {
            self.last_used.fetch_max(tick, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy)]
enum Step {
    /// The whole edge matched.
    Full(usize),
    /// The match ends at `marks[mark]` of the node.
    Partial(usize, usize),
}

#[derive(Default)]
struct Hit {
    /// The namespace root, if it existed.
    root: Option<usize>,
    /// Tree version the steps are valid for.
    version: u64,
    steps: Vec<Step>,
    bytes: usize,
    tokens: usize,
    pieces: Vec<(Arc<[u32]>, usize)>,
}

struct Tree {
    nodes: Vec<Option<Box<Node>>>,
    free: Vec<usize>,
    roots: hashbrown::HashMap<Namespace, usize>,
    /// First child per (parent, first-segment key); collisions chain
    /// through `Node::next_same_key`.
    edges: hashbrown::HashMap<EdgeKey, usize>,
    /// Leaves by last use. Lazy: an entry may name a node that is gone, is
    /// no longer a leaf, or was used since; every leaf has an entry.
    leaves: BinaryHeap<Reverse<(u64, usize)>>,
    live: usize,
    /// Heap bytes owned by live nodes.
    content_bytes: usize,
    /// Bumped by every structural change, so a lookup's steps can be reused.
    version: u64,
    strategy: L1HashStrategy,
}

impl Tree {
    fn new(strategy: L1HashStrategy) -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            roots: hashbrown::HashMap::new(),
            edges: hashbrown::HashMap::new(),
            leaves: BinaryHeap::new(),
            live: 0,
            content_bytes: 0,
            version: 0,
            strategy,
        }
    }

    /// Estimated heap bytes: node contents plus the capacity of every
    /// container, so slack left behind by evictions is charged too.
    fn bytes(&self) -> usize {
        self.content_bytes
            + self.nodes.capacity() * size_of::<Option<Box<Node>>>()
            + self.free.capacity() * size_of::<usize>()
            + self.leaves.capacity() * size_of::<Reverse<(u64, usize)>>()
            + self.edges.allocation_size()
            + self.roots.allocation_size()
    }

    fn key(&self, segment: &str) -> SegmentKey {
        segment_key(segment, self.strategy)
    }

    fn node(&self, id: usize) -> &Node {
        self.nodes[id].as_deref().expect("live L1 node")
    }

    fn node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes[id].as_deref_mut().expect("live L1 node")
    }

    fn alloc(&mut self, node: Node) -> usize {
        self.version += 1;
        self.live += 1;
        self.content_bytes += node.bytes();
        let node = Some(Box::new(node));
        match self.free.pop() {
            Some(id) => {
                self.nodes[id] = node;
                id
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        }
    }

    fn release(&mut self, id: usize) -> Box<Node> {
        self.version += 1;
        let node = self.nodes[id].take().expect("live L1 node");
        self.free.push(id);
        self.live -= 1;
        self.content_bytes -= node.bytes();
        node
    }

    fn edge(&self, parent: usize, child: usize) -> EdgeKey {
        EdgeKey {
            parent,
            segment: self.key(self.node(child).first_segment()),
        }
    }

    fn child(&self, parent: usize, segment: &str) -> Option<usize> {
        let key = EdgeKey {
            parent,
            segment: self.key(segment),
        };
        let mut id = *self.edges.get(&key)?;
        while self.node(id).first_segment() != segment {
            id = self.node(id).next_same_key;
            if id == NO_NODE {
                return None;
            }
        }
        Some(id)
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let key = self.edge(parent, child);
        let next = self.edges.insert(key, child).unwrap_or(NO_NODE);
        self.node_mut(child).next_same_key = next;
        self.node_mut(parent).child_count += 1;
    }

    fn unlink(&mut self, parent: usize, child: usize) {
        let key = self.edge(parent, child);
        let next = self.node(child).next_same_key;
        if self.edges[&key] == child {
            if next == NO_NODE {
                self.edges.remove(&key);
            } else {
                self.edges.insert(key, next);
            }
        } else {
            let mut prev = self.edges[&key];
            while self.node(prev).next_same_key != child {
                prev = self.node(prev).next_same_key;
            }
            self.node_mut(prev).next_same_key = next;
        }
        self.node_mut(child).next_same_key = NO_NODE;
        self.node_mut(parent).child_count -= 1;
    }

    /// Follow `text` from `root`, comparing edge text directly.
    /// `first_cut(offset)` returns the end of the first segment of
    /// `text[offset..]` and is only asked at cut positions; it is used to
    /// pick the child. Returns the steps and the matched bytes, which end at
    /// a cut.
    fn walk(
        &self,
        root: usize,
        text: &str,
        mut first_cut: impl FnMut(usize) -> Option<usize>,
    ) -> (Vec<Step>, usize) {
        let mut steps = Vec::new();
        let (mut node, mut offset) = (root, 0);
        while let Some(end) = first_cut(offset) {
            let Some(child) = self.child(node, &text[offset..end]) else {
                break;
            };
            let edge = self.node(child);
            let rest = &text.as_bytes()[offset..];
            if rest.starts_with(edge.text.as_bytes()) {
                steps.push(Step::Full(child));
                offset += edge.text.len();
                node = child;
                continue;
            }
            let common = common_prefix(edge.text.as_bytes(), rest);
            // The first segment matched, so a partial match has a mark.
            let mark = edge
                .marks
                .partition_point(|&(byte, _)| byte as usize <= common)
                - 1;
            offset += edge.marks[mark].0 as usize;
            steps.push(Step::Partial(child, mark));
            break;
        }
        (steps, offset)
    }

    fn lookup(
        &self,
        root: usize,
        text: &str,
        first_cut: impl FnMut(usize) -> Option<usize>,
        tick: u64,
    ) -> Hit {
        let (steps, _) = self.walk(root, text, first_cut);
        let mut hit = Hit {
            root: Some(root),
            version: self.version,
            ..Default::default()
        };
        for &step in &steps {
            let (id, bytes, tokens) = match step {
                Step::Full(id) => {
                    let node = self.node(id);
                    (id, node.text.len(), node.ids.len())
                }
                Step::Partial(id, mark) => {
                    let (byte, token) = self.node(id).marks[mark];
                    (id, byte as usize, token as usize)
                }
            };
            let node = self.node(id);
            node.touch(tick);
            hit.pieces.push((node.ids.clone(), tokens));
            hit.bytes += bytes;
            hit.tokens += tokens;
        }
        hit.steps = steps;
        hit
    }

    fn matched_tokens(&self, steps: &[Step]) -> usize {
        steps
            .iter()
            .map(|&step| match step {
                Step::Full(id) => self.node(id).ids.len(),
                Step::Partial(id, mark) => self.node(id).marks[mark].1 as usize,
            })
            .sum()
    }

    fn root(&mut self, namespace: &Namespace, tick: u64) -> usize {
        if let Some(&root) = self.roots.get(namespace) {
            return root;
        }
        let root = self.alloc(Node::new(
            NO_NODE,
            Box::from(""),
            Arc::from([]),
            Box::from([]),
            tick,
            Some(namespace.clone()),
        ));
        self.roots.insert(namespace.clone(), root);
        root
    }

    /// Content bytes that splitting `id` at `marks[mark]` adds.
    fn split_bytes(&self, id: usize, mark: usize) -> usize {
        let node = self.node(id);
        let (byte, token) = node.marks[mark];
        let (byte, token) = (byte as usize, token as usize);
        let upper = node_bytes(byte, token, mark, None);
        let lower = node_bytes(
            node.text.len() - byte,
            node.ids.len() - token,
            node.marks.len() - mark - 1,
            None,
        );
        (upper + lower).saturating_sub(node.bytes())
    }

    /// Split `id` at `marks[mark]`; returns the new upper node.
    fn split(&mut self, id: usize, mark: usize) -> usize {
        let parent = self.node(id).parent;
        self.unlink(parent, id);
        let old_bytes = self.node(id).bytes();
        let node = self.node_mut(id);
        let (byte, token) = node.marks[mark];
        let (byte, token) = (byte as usize, token as usize);
        let upper = Node::new(
            parent,
            Box::from(&node.text[..byte]),
            Arc::from(&node.ids[..token]),
            Box::from(&node.marks[..mark]),
            node.last_used.load(Ordering::Relaxed),
            None,
        );
        node.text = Box::from(&node.text[byte..]);
        node.ids = Arc::from(&node.ids[token..]);
        node.marks = node.marks[mark + 1..]
            .iter()
            .map(|&(b, t)| (b - byte as u32, t - token as u32))
            .collect();
        let new_bytes = node.bytes();
        self.content_bytes = self.content_bytes - old_bytes + new_bytes;
        // The upper node keeps the old first segment, hence the same key.
        let upper = self.alloc(upper);
        self.add_child(parent, upper);
        self.node_mut(id).parent = upper;
        self.add_child(upper, id);
        upper
    }

    fn push_leaf(&mut self, id: usize) {
        let tick = self.node(id).last_used.load(Ordering::Relaxed);
        self.leaves.push(Reverse((tick, id)));
    }

    /// Remove a leaf. Single-child chains are kept: merging them would copy
    /// whole edges under the write lock.
    fn remove_leaf(&mut self, id: usize) {
        let parent = self.node(id).parent;
        self.unlink(parent, id);
        self.release(id);
        let parent_node = self.node(parent);
        if parent_node.child_count > 0 {
            return;
        }
        if parent_node.namespace.is_some() {
            if let Some(namespace) = self.release(parent).namespace {
                self.roots.remove(&namespace);
            }
        } else {
            // Chains shrink from the end: a parent that became a leaf is next.
            self.push_leaf(parent);
        }
    }

    /// Evict least-recently-used leaves until both budgets hold. Each step
    /// is O(log n): stale heap entries are dropped, or re-queued at the
    /// node's current tick, when they reach the top.
    fn evict(&mut self, max_bytes: usize, max_nodes: usize) -> u64 {
        let mut evicted = 0;
        while self.live > 0 && (self.live > max_nodes || self.bytes() > max_bytes) {
            let Some(Reverse((tick, id))) = self.leaves.pop() else {
                debug_assert!(false, "every leaf has a heap entry");
                break;
            };
            let Some(node) = self.nodes.get(id).and_then(Option::as_deref) else {
                continue;
            };
            if !node.is_leaf() {
                continue;
            }
            let now = node.last_used.load(Ordering::Relaxed);
            if now != tick {
                self.leaves.push(Reverse((now, id)));
                continue;
            }
            self.remove_leaf(id);
            evicted += 1;
            self.shrink_sparse();
        }
        if self.live == 0 {
            self.release_storage();
        } else if self.leaves.len() > 2 * self.live + 16 {
            self.compact_leaves();
        }
        evicted
    }

    /// Release container capacity once it is mostly unused, so evictions
    /// also return the slack `bytes` charges. A shrink only follows enough
    /// removals to pay for it.
    fn shrink_sparse(&mut self) {
        let sparse = |len: usize, capacity: usize| capacity > 64 && len * 4 < capacity;
        if sparse(self.edges.len(), self.edges.capacity()) {
            self.edges.shrink_to_fit();
        }
        if sparse(self.roots.len(), self.roots.capacity()) {
            self.roots.shrink_to_fit();
        }
        if sparse(self.leaves.len(), self.leaves.capacity()) {
            self.leaves.shrink_to_fit();
        }
        if sparse(self.free.len(), self.free.capacity()) {
            self.free.shrink_to_fit();
        }
    }

    /// Keep one entry per live leaf, at its current tick.
    fn compact_leaves(&mut self) {
        let nodes = &self.nodes;
        let mut entries: Vec<(usize, u64)> = self
            .leaves
            .drain()
            .filter_map(|Reverse((_, id))| {
                let node = nodes.get(id)?.as_deref()?;
                node.is_leaf()
                    .then(|| (id, node.last_used.load(Ordering::Relaxed)))
            })
            .collect();
        entries.sort_unstable();
        entries.dedup_by_key(|&mut (id, _)| id);
        self.leaves = entries
            .into_iter()
            .map(|(id, tick)| Reverse((tick, id)))
            .collect();
    }

    fn release_storage(&mut self) {
        self.version += 1;
        self.nodes = Vec::new();
        self.free = Vec::new();
        self.leaves = BinaryHeap::new();
        self.edges = hashbrown::HashMap::new();
        self.roots = hashbrown::HashMap::new();
    }
}

/// Occupancy summed over every live L1 cache. The lock orders gauge
/// publications, so a scrape sees one snapshot of both totals.
struct Occupancy {
    nodes: usize,
    bytes: usize,
}

static OCCUPANCY: Mutex<Occupancy> = Mutex::new(Occupancy { nodes: 0, bytes: 0 });

/// Apply one cache's change and publish both gauges. Insert calls this while
/// holding the tree write lock; drop calls it with exclusive access to that
/// cache. Lock order is the tree, then `OCCUPANCY`.
fn publish_occupancy(nodes_delta: isize, bytes_delta: isize) {
    if nodes_delta == 0 && bytes_delta == 0 {
        return;
    }
    let mut total = OCCUPANCY.lock();
    total.nodes = total.nodes.saturating_add_signed(nodes_delta);
    total.bytes = total.bytes.saturating_add_signed(bytes_delta);
    TokenizerMetrics::set_l1_nodes(total.nodes);
    TokenizerMetrics::set_l1_bytes(total.bytes);
}

/// Shared L1 tree with one namespace per (tokenizer instance, cache_salt).
pub(crate) struct L1Cache {
    config: L1CacheConfig,
    tree: RwLock<Tree>,
    /// LRU clock. Only inserts advance it, so lookups never write it and
    /// write a node at most once between inserts.
    tick: AtomicU64,
    lookups: AtomicU64,
    hits: AtomicU64,
    cached_tokens: AtomicU64,
    encoded_tokens: AtomicU64,
    evictions: AtomicU64,
    oversized: AtomicU64,
    verify_mismatches: AtomicU64,
    bypasses: AtomicU64,
}

impl L1Cache {
    pub(crate) fn new(config: L1CacheConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            tree: RwLock::new(Tree::new(config.hash_strategy)),
            tick: AtomicU64::new(0),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            encoded_tokens: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
            verify_mismatches: AtomicU64::new(0),
            bypasses: AtomicU64::new(0),
        })
    }

    pub(crate) fn stats(&self) -> L1CacheStats {
        let (nodes, bytes) = {
            let tree = self.tree.read();
            (tree.live, tree.bytes())
        };
        L1CacheStats {
            lookups: self.lookups.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            encoded_tokens: self.encoded_tokens.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
            verify_mismatches: self.verify_mismatches.load(Ordering::Relaxed),
            bypasses: self.bypasses.load(Ordering::Relaxed),
            nodes,
            bytes,
        }
    }

    fn lookup(
        &self,
        namespace: &Namespace,
        text: &str,
        first_cut: impl FnMut(usize) -> Option<usize>,
    ) -> Hit {
        let tick = self.tick.load(Ordering::Relaxed);
        let tree = self.tree.read();
        match tree.roots.get(namespace) {
            Some(&root) => tree.lookup(root, text, first_cut, tick),
            None => Hit {
                version: tree.version,
                ..Default::default()
            },
        }
    }

    /// Store the cuts after `hit`. `bounds` are the absolute (byte, token)
    /// offsets of cuts after `hit.bytes`, in order.
    fn insert(
        &self,
        namespace: &Namespace,
        text: &str,
        ids: &[u32],
        hit: Hit,
        bounds: &[(usize, usize)],
        first_cut: impl FnMut(usize) -> Option<usize>,
    ) {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        let mut tree = self.tree.write();
        let (before_nodes, before_bytes) = (tree.live, tree.bytes());
        let (root, steps, offset, start_token) = if tree.version == hit.version {
            (hit.root, hit.steps, hit.bytes, hit.tokens)
        } else if let Some(&root) = tree.roots.get(namespace) {
            let (steps, offset) = tree.walk(root, text, first_cut);
            let tokens = tree.matched_tokens(&steps);
            (Some(root), steps, offset, tokens)
        } else {
            (None, Vec::new(), 0, 0)
        };
        // An eviction since the lookup removed part of the path; the cuts
        // in between have no token offsets here, so leave it to a later
        // request.
        if offset >= hit.bytes {
            let start = bounds.partition_point(|&(byte, _)| byte <= offset);
            let end = bounds.len().min(start + self.config.max_insert_segments);
            let fresh = &bounds[start..end];
            let salt = namespace.salt.as_ref().map(|salt| salt.len());
            let fixed = if root.is_none() {
                node_bytes(0, 0, 0, salt)
            } else {
                0
            } + match steps.last() {
                Some(&Step::Partial(id, mark)) => tree.split_bytes(id, mark),
                _ => 0,
            };
            // Cost grows with the number of segments; keep the longest run
            // that fits.
            let cost = |i: usize| {
                let (byte, token) = fresh[i];
                fixed + node_bytes(byte - offset, token - start_token, i, None)
            };
            let (mut stored, mut over) = (0, fresh.len());
            while stored < over {
                let mid = (stored + over) / 2;
                if cost(mid) <= self.config.max_insert_bytes {
                    stored = mid + 1;
                } else {
                    over = mid;
                }
            }
            if stored == 0 && !fresh.is_empty() {
                self.oversized.fetch_add(1, Ordering::Relaxed);
            }
            if stored > 0 {
                let mut parent = root.unwrap_or_else(|| tree.root(namespace, tick));
                for &step in &steps {
                    match step {
                        Step::Full(id) => {
                            tree.node(id).touch(tick);
                            parent = id;
                        }
                        Step::Partial(id, mark) => parent = tree.split(id, mark),
                    }
                }
                let (end_byte, end_token) = fresh[stored - 1];
                let marks: Box<[(u32, u32)]> = fresh[..stored - 1]
                    .iter()
                    .map(|&(b, t)| ((b - offset) as u32, (t - start_token) as u32))
                    .collect();
                // Growth appends a child; existing edges are never copied.
                let leaf = tree.alloc(Node::new(
                    parent,
                    Box::from(&text[offset..end_byte]),
                    Arc::from(&ids[start_token..end_token]),
                    marks,
                    tick,
                    None,
                ));
                tree.add_child(parent, leaf);
                tree.push_leaf(leaf);
            }
        }
        let evicted = tree.evict(self.config.max_bytes, self.config.max_nodes);
        if evicted > 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
            TokenizerMetrics::record_l1_evictions(evicted);
        }
        publish_occupancy(
            tree.live as isize - before_nodes as isize,
            tree.bytes() as isize - before_bytes as isize,
        );
    }
}

impl Drop for L1Cache {
    fn drop(&mut self) {
        let tree = self.tree.get_mut();
        publish_occupancy(-(tree.live as isize), -(tree.bytes() as isize));
    }
}

/// Per-model L1 handle used by `PromptEncoder`.
#[derive(Clone)]
pub(crate) struct L1Encoder {
    cache: Arc<L1Cache>,
    matcher: Arc<BoundaryMatcher>,
    model_id: Arc<str>,
    disabled: Arc<AtomicBool>,
    mismatch_warning: Arc<Once>,
    segment_warning: Arc<Once>,
}

impl L1Encoder {
    pub(crate) fn new(model_id: &str, cache: Arc<L1Cache>, matcher: BoundaryMatcher) -> Self {
        Self {
            cache,
            matcher: Arc::new(matcher),
            model_id: Arc::from(model_id),
            disabled: Arc::new(AtomicBool::new(false)),
            mismatch_warning: Arc::new(Once::new()),
            segment_warning: Arc::new(Once::new()),
        }
    }

    /// Encode with L1, or `None` if the prompt has no boundary or L1 was
    /// disabled. Returns the IDs and how many came from the tree.
    pub(crate) fn encode(
        &self,
        tokenizer: &dyn Tokenizer,
        instance: u64,
        text: &str,
        cache_salt: Option<&str>,
    ) -> vllm_tokenizer::Result<Option<(Vec<u32>, usize)>> {
        if self.disabled.load(Ordering::Relaxed) {
            self.record_bypass(L1Bypass::Disabled);
            return Ok(None);
        }
        let namespace = Namespace {
            instance,
            salt: cache_salt.map(Arc::<str>::from),
        };
        let matcher = &*self.matcher;
        let first_cut = |from: usize| matcher.first_cut(text, from);
        let hit = self.cache.lookup(&namespace, text, first_cut);
        // Every cut up to the match is already in the tree; scan the rest.
        let cuts = matcher.cuts(text, hit.bytes);
        if hit.bytes == 0 && cuts.is_empty() {
            self.record_bypass(L1Bypass::NoBoundary);
            return Ok(None);
        }
        // The tail starts at a cut, so one encode of it has the same IDs as
        // encoding each segment.
        let tail = &text[hit.bytes..];
        let tail_ids = if tail.is_empty() {
            Vec::new()
        } else {
            tokenizer.encode(tail, false)?
        };
        let bounds = self.tail_bounds(tokenizer, text, &hit, &cuts, &tail_ids)?;
        let mut ids = Vec::with_capacity(hit.tokens + tail_ids.len());
        for (piece, take) in &hit.pieces {
            ids.extend_from_slice(&piece[..*take]);
        }
        ids.extend(tail_ids);
        if self.cache.config.verify {
            let full = tokenizer.encode(text, false)?;
            if full != ids {
                self.disabled.store(true, Ordering::Relaxed);
                self.cache.verify_mismatches.fetch_add(1, Ordering::Relaxed);
                TokenizerMetrics::record_l1_verify_mismatch();
                let first = full.iter().zip(&ids).take_while(|(a, b)| a == b).count();
                self.mismatch_warning.call_once(|| {
                    tracing::error!(
                        model_id = %self.model_id,
                        first_mismatch = first,
                        full_len = full.len(),
                        l1_len = ids.len(),
                        "L1 verify mismatch; L1 disabled for this model"
                    );
                });
                return Ok(Some((full, 0)));
            }
        }
        let cached = hit.tokens;
        self.cache.lookups.fetch_add(1, Ordering::Relaxed);
        self.cache
            .cached_tokens
            .fetch_add(cached as u64, Ordering::Relaxed);
        self.cache
            .encoded_tokens
            .fetch_add((ids.len() - cached) as u64, Ordering::Relaxed);
        TokenizerMetrics::record_l1_lookup(cached, ids.len() - cached);
        if cached > 0 {
            self.cache.hits.fetch_add(1, Ordering::Relaxed);
        }
        if !bounds.is_empty() {
            self.cache
                .insert(&namespace, text, &ids, hit, &bounds, first_cut);
        }
        Ok(Some((ids, cached)))
    }

    /// Absolute (byte, token) offsets of the tail's `cuts`, read from the
    /// tail encoding when boundary IDs line up. Otherwise the segments are
    /// encoded one at a time, as far as insert could store them.
    fn tail_bounds(
        &self,
        tokenizer: &dyn Tokenizer,
        text: &str,
        hit: &Hit,
        cuts: &[(usize, usize)],
        tail_ids: &[u32],
    ) -> vllm_tokenizer::Result<Vec<(usize, usize)>> {
        if let Some(offsets) = self.matcher.token_offsets(cuts, tail_ids) {
            return Ok(cuts
                .iter()
                .zip(offsets)
                .map(|(&(byte, _), token)| (byte, hit.tokens + token))
                .collect());
        }
        self.segment_warning.call_once(|| {
            tracing::debug!(
                model_id = %self.model_id,
                "boundary token IDs do not line up with the encoding; L1 encodes segments one at a time"
            );
        });
        let config = &self.cache.config;
        let (mut from, mut tokens) = (hit.bytes, 0);
        let mut bounds = Vec::new();
        for &(byte, _) in cuts.iter().take(config.max_insert_segments) {
            if byte - hit.bytes > config.max_insert_bytes {
                break;
            }
            let piece = tokenizer.encode(&text[from..byte], false)?;
            // Never store a segment that disagrees with the tail encoding.
            if tail_ids.get(tokens..tokens + piece.len()) != Some(&piece[..]) {
                break;
            }
            tokens += piece.len();
            from = byte;
            bounds.push((byte, hit.tokens + tokens));
        }
        Ok(bounds)
    }

    pub(crate) fn record_bypass(&self, reason: L1Bypass) {
        self.cache.bypasses.fetch_add(1, Ordering::Relaxed);
        TokenizerMetrics::record_l1_bypass(reason.as_str());
    }
}

pub(crate) fn config_from_env() -> Result<Option<L1CacheConfig>> {
    parse_config(|name| std::env::var(name).ok())
}

fn parse_flag(name: &str, raw: Option<String>) -> Result<bool> {
    match raw.as_deref() {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        _ => bail!("{name} must be 0, 1, false or true"),
    }
}

fn parse_hash_strategy(raw: Option<String>) -> Result<L1HashStrategy> {
    match raw.as_deref() {
        None | Some("first_last_64" | "first-last-64" | "first_last" | "first-last") => {
            Ok(L1HashStrategy::FirstLast64)
        }
        Some("full") | Some("full_segment") | Some("full-segment") => Ok(L1HashStrategy::Full),
        Some(raw) => bail!("VLLM_ROUTER_L1_HASH_STRATEGY must be first_last_64 or full, got {raw}"),
    }
}

fn parse_config(read: impl Fn(&str) -> Option<String>) -> Result<Option<L1CacheConfig>> {
    if !parse_flag("VLLM_ROUTER_L1_CACHE", read("VLLM_ROUTER_L1_CACHE"))? {
        return Ok(None);
    }
    let mut config = L1CacheConfig {
        verify: parse_flag("VLLM_ROUTER_L1_VERIFY", read("VLLM_ROUTER_L1_VERIFY"))?,
        hash_strategy: parse_hash_strategy(read("VLLM_ROUTER_L1_HASH_STRATEGY"))?,
        ..Default::default()
    };
    let explicit_insert = read("VLLM_ROUTER_L1_MAX_INSERT_BYTES").is_some();
    for (name, value) in [
        ("VLLM_ROUTER_L1_MAX_BYTES", &mut config.max_bytes),
        ("VLLM_ROUTER_L1_MAX_NODES", &mut config.max_nodes),
        (
            "VLLM_ROUTER_L1_MAX_INSERT_BYTES",
            &mut config.max_insert_bytes,
        ),
        (
            "VLLM_ROUTER_L1_MAX_INSERT_SEGMENTS",
            &mut config.max_insert_segments,
        ),
    ] {
        if let Some(raw) = read(name) {
            *value = raw
                .parse()
                .with_context(|| format!("invalid {name}: {raw}"))?;
        }
    }
    if !explicit_insert {
        config.max_insert_bytes = config.max_insert_bytes.min(config.max_bytes);
    }
    config.validate()?;
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::{
        Counter, Gauge, GaugeFn, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;
    use vllm_tokenizer::HuggingFaceTokenizer;

    fn spec(content: &str) -> AddedTokenSpec {
        AddedTokenSpec {
            content: content.into(),
            lstrip: false,
            rstrip: false,
            single_word: false,
            normalized: false,
        }
    }

    fn token_set(tokens: Vec<AddedTokenSpec>, identity_normalizer: bool) -> AddedTokenSet {
        AddedTokenSet {
            tokens,
            identity_normalizer,
        }
    }

    fn eligible_in(set: &AddedTokenSet) -> Vec<String> {
        BoundaryMatcher::new(set)
            .map(|m| m.tokens().to_vec())
            .unwrap_or_default()
    }

    fn eligible(tokens: &[&str]) -> Vec<String> {
        eligible_in(&token_set(tokens.iter().map(|t| spec(t)).collect(), true))
    }

    fn normalized(content: &str) -> AddedTokenSpec {
        AddedTokenSpec {
            normalized: true,
            ..spec(content)
        }
    }

    #[test]
    fn boundary_tokens_exclude_overlaps_and_flags() {
        let qwen = [
            "<|im_start|>",
            "<|im_end|>",
            "<tool_call>",
            "</tool_call>",
            "<think>",
            "</think>",
            "<tts_text_bos>",
            "<tts_text_bos_single>",
        ];
        assert_eq!(eligible(&qwen).len(), qwen.len());
        // A longer token containing `t` before its end can win in `P + S`.
        assert_eq!(eligible(&["<a>", "<a>b"]), ["<a>b"]);
        assert_eq!(eligible(&["b", "abc"]), ["abc"]);
        // Ending inside or at the end of another token is decided by the
        // matcher, which sees every added token.
        assert_eq!(eligible(&[">x", "a>x"]), [">x", "a>x"]);
        assert_eq!(eligible(&["xy", "ax"]), ["ax", "xy"]);
        assert_eq!(eligible(&["aa", " <x>", "<x> "]), ["aa"]);
        for flag in 0..3 {
            let mut token = spec("<x>");
            match flag {
                0 => token.lstrip = true,
                1 => token.rstrip = true,
                _ => token.single_word = true,
            }
            assert!(BoundaryMatcher::new(&token_set(vec![token], true)).is_err());
        }
        // A single_word token after a word character depends on the cut.
        let word = AddedTokenSpec {
            single_word: true,
            ..spec("w")
        };
        assert_eq!(
            eligible_in(&token_set(vec![spec("<x>"), spec("<x1"), word], true)),
            ["<x>"]
        );
    }

    #[test]
    fn single_word_guard_matches_hugging_face_word_chars() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

        let vocab = [("[UNK]".to_string(), 0)].into_iter().collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = HfTokenizer::new(model);
        // One candidate boundary per last character, plus a single_word token
        // that can follow the cut.
        let boundaries: Vec<String> = (0x21u32..0x3000)
            .filter_map(char::from_u32)
            .filter(|c| !c.is_whitespace() && !c.is_control())
            .map(|c| format!("§{c}"))
            .collect();
        let mut added: Vec<AddedToken> = boundaries
            .iter()
            .map(|b| AddedToken::from(b.as_str(), false).normalized(false))
            .collect();
        added.push(
            AddedToken::from("rareword", false)
                .single_word(true)
                .normalized(false),
        );
        tokenizer.add_tokens(&added);

        let json: Value = serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
        let set = token_set(added_tokens(&json["added_tokens"], None), true);
        let verdicts: HashMap<&str, Option<&str>> = set
            .tokens
            .iter()
            .map(|t| t.content.as_str())
            .zip(boundary_rejections(&set))
            .collect();

        let encode = |text: &str| tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let suffix = encode("rareword");
        let mut breaking = 0;
        for boundary in &boundaries {
            let mut split = encode(boundary);
            split.extend(&suffix);
            if encode(&format!("{boundary}rareword")) != split {
                breaking += 1;
                assert_eq!(
                    verdicts[boundary.as_str()],
                    Some("a single_word token could follow"),
                    "{boundary:?} changes the IDs but is a boundary"
                );
            }
        }
        assert!(breaking > 0);
        // Combining acute accent: `\w` but not alphanumeric.
        assert!(verdicts["§\u{301}"].is_some());
        assert_eq!(encode("§\u{301}rareword"), [encode("§\u{301}")[0], 0]);
    }

    #[test]
    fn normalized_tokens_need_identity_normalizer_and_separate_groups() {
        let separate = vec![spec("aa"), normalized("<n>")];
        assert_eq!(
            eligible_in(&token_set(separate.clone(), true)),
            ["<n>", "aa"]
        );
        assert_eq!(eligible_in(&token_set(separate, false)), ["aa"]);
        // `s>` overlaps `<s>`: backends may split differently, so normalized
        // tokens are out and the strict rules apply (no self overlap).
        let mixed = vec![spec("<s>"), spec("aa"), normalized("s>")];
        assert_eq!(eligible_in(&token_set(mixed, true)), ["<s>"]);
        assert!(is_identity_normalizer(&Value::Null));
        assert!(is_identity_normalizer(
            &json!({"type": "Sequence", "normalizers": []})
        ));
        assert!(!is_identity_normalizer(&json!({"type": "NFC"})));
    }

    #[test]
    fn matcher_cuts_only_where_the_tokenizer_splits() {
        let matcher =
            BoundaryMatcher::new(&token_set(vec![spec(">x"), spec("a>x")], true)).unwrap();
        assert_eq!(matcher.boundaries("a>x b>x"), [3, 7]);
        // A non-boundary `a<s` hides the first `<s>` from the tokenizer.
        let matcher =
            BoundaryMatcher::build(vec!["<s>".into(), "a<s".into()], vec!["<s>".into()]).unwrap();
        assert_eq!(matcher.boundaries("a<s> <s>"), [8]);
    }

    #[test]
    fn added_tokens_merge_config_and_default_normalized() {
        let json = json!([{"id": 1, "content": "<a>", "special": true, "normalized": false}]);
        let decoder = json!({
            "1": {"content": "<dup>", "special": true},
            "2": {"content": "<b>", "special": true},
            "3": {"content": "plain"}
        });
        let mut tokens = added_tokens(&json, None);
        tokens.extend(added_tokens(&decoder, Some(&[1])));
        let names: Vec<_> = tokens.iter().map(|t| t.content.as_str()).collect();
        assert_eq!(names, ["<a>", "<b>", "plain"]);
        assert!(!tokens[1].normalized);
        // "plain" omits `normalized`. vllm-tokenizer loads that as false.
        assert!(!tokens[2].normalized);
    }

    #[test]
    fn decoder_ids_that_do_not_fit_in_u32_are_not_boundaries() {
        // vllm-tokenizer 0.29 parses the key as u32 and skips a parse failure.
        // "4294967296" is 2^32. Admitting its content would cut where the
        // tokenizer still sees ordinary BPE.
        let tokens = added_tokens(
            &json!({
                "4294967296": {"content": "X"},
                "4294967295": {"content": "<max>"},
                "nope": {"content": "Y"}
            }),
            None,
        );
        let names: Vec<_> = tokens.iter().map(|token| token.content.as_str()).collect();
        assert_eq!(names, ["<max>"]);
    }

    #[test]
    fn missing_normalized_does_not_share_a_group_with_an_explicit_token() {
        // "ab" is normalized and "bc" omits the field. Treating the omission
        // as normalized puts both in one group, so a one-pass cut after "ab"
        // disagrees with Hugging Face's two-pass encode of "abc".
        let tokens = added_tokens(
            &json!([
                {"content": "ab", "normalized": true},
                {"content": "bc"}
            ]),
            None,
        );
        assert!(tokens[0].normalized);
        assert!(!tokens[1].normalized);
        assert!(eligible_in(&token_set(tokens, true)).is_empty());
    }

    #[test]
    fn rejects_unsupported_tokenizer_configs() {
        let base =
            json!({"model": {"type": "BPE", "dropout": null}, "truncation": null, "padding": null});
        assert!(check_tokenizer_json(&base).is_ok());
        for (path, value) in [
            ("truncation", json!({"max_length": 8})),
            ("padding", json!({"strategy": "BatchLongest"})),
            ("model", json!({"type": "BPE", "dropout": 0.1})),
            (
                "pre_tokenizer",
                json!({"type": "Sequence", "pretokenizers": [{"type": "Metaspace", "prepend_scheme": "first"}]}),
            ),
        ] {
            let mut json = base.clone();
            json[path] = value;
            assert!(check_tokenizer_json(&json).is_err(), "{path}");
        }
        let mut always = base.clone();
        always["pre_tokenizer"] = json!({"type": "Metaspace", "prepend_scheme": "always"});
        assert!(check_tokenizer_json(&always).is_ok());
    }

    fn fixture_tokenizer() -> HuggingFaceTokenizer {
        HuggingFaceTokenizer::new(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/rust/fixtures/tokenizer/byte_level_bpe.json"
        )))
        .unwrap()
    }

    fn encoder(config: L1CacheConfig) -> (L1Encoder, Arc<L1Cache>) {
        let cache = Arc::new(L1Cache::new(config).unwrap());
        let matcher = BoundaryMatcher::new(&token_set(vec![spec("<s>"), spec("</s>")], true))
            .unwrap()
            .with_token_ids(&fixture_tokenizer());
        (L1Encoder::new("test-model", cache.clone(), matcher), cache)
    }

    fn check(encoder: &L1Encoder, tokenizer: &dyn Tokenizer, text: &str) -> usize {
        let (ids, cached) = encoder
            .encode(tokenizer, 0, text, None)
            .unwrap()
            .expect("boundaries");
        assert_eq!(ids, tokenizer.encode(text, false).unwrap(), "{text:?}");
        cached
    }

    #[test]
    fn tree_reuses_longest_boundary_prefix_and_splits_edges() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let a = "<s>system hi</s><s>user one</s><s>assistant";
        assert_eq!(check(&encoder, &tokenizer, a), 0);
        let full = tokenizer.encode(a, false).unwrap();
        // Same prompt: everything up to the last boundary is reused.
        let head = tokenizer
            .encode("<s>system hi</s><s>user one</s><s>", false)
            .unwrap();
        assert_eq!(check(&encoder, &tokenizer, a), head.len());
        assert!(head.len() < full.len());
        // Growing conversation reuses the whole old prefix.
        let b = "<s>system hi</s><s>user one</s><s>assistant ok</s><s>user two</s><s>assistant";
        assert_eq!(check(&encoder, &tokenizer, b), head.len());
        // An edited second message stops at the boundary before it (split).
        let c = "<s>system hi</s><s>user ONE</s><s>assistant";
        let first = tokenizer.encode("<s>system hi</s><s>", false).unwrap();
        assert_eq!(check(&encoder, &tokenizer, c), first.len());
        // Both branches still hit after the split.
        assert!(check(&encoder, &tokenizer, b) > head.len());
        assert!(check(&encoder, &tokenizer, c) > first.len());
        // Text without boundaries, a literal partial token, and trailing text.
        assert!(encoder
            .encode(&tokenizer, 0, "plain", None)
            .unwrap()
            .is_none());
        check(&encoder, &tokenizer, "<s>system hi</s><s>user <s</s>x");
        check(&encoder, &tokenizer, "<s>system hi</s>");
        check(&encoder, &tokenizer, "<s>");
        let stats = cache.stats();
        assert!(stats.hits >= 5 && stats.nodes > 0 && stats.bytes > 0);
    }

    #[test]
    fn growth_appends_a_child_without_copying() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let edges = || -> Vec<(usize, Arc<[u32]>)> {
            let tree = cache.tree.read();
            (0..tree.nodes.len())
                .filter_map(|id| tree.nodes[id].as_ref().map(|node| (id, node.ids.clone())))
                .collect()
        };
        let mut text = String::from("<s>system hi</s>");
        check(&encoder, &tokenizer, &format!("{text}<s>"));
        for turn in 0..5 {
            let before = edges();
            text.push_str(&format!("<s>user {turn}</s><s>assistant {turn}</s>"));
            let cached = check(&encoder, &tokenizer, &format!("{text}<s>"));
            assert!(cached > 0);
            let after = edges();
            assert_eq!(after.len(), before.len() + 1, "one new edge per turn");
            for (id, ids) in &before {
                let (_, now) = after.iter().find(|(other, _)| other == id).unwrap();
                assert!(Arc::ptr_eq(ids, now), "existing edges are untouched");
            }
            check_invariants(&cache);
        }
    }

    #[test]
    fn namespaces_are_isolated() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let text = "<s>system hi</s><s>user</s>";
        for (instance, salt) in [(0, None), (1, None), (0, Some("a")), (0, Some(""))] {
            let (_, cached) = encoder
                .encode(&tokenizer, instance, text, salt)
                .unwrap()
                .unwrap();
            assert_eq!(cached, 0);
            let (_, cached) = encoder
                .encode(&tokenizer, instance, text, salt)
                .unwrap()
                .unwrap();
            assert!(cached > 0);
        }
        assert_eq!(cache.stats().hits, 4);
    }

    #[test]
    fn bytes_count_node_contents_and_container_capacity() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        check(&encoder, &tokenizer, "<s>system hi</s><s>user</s>");
        let tree = cache.tree.read();
        let containers = tree.nodes.capacity() * size_of::<Option<Box<Node>>>()
            + tree.free.capacity() * size_of::<usize>()
            + tree.leaves.capacity() * size_of::<Reverse<(u64, usize)>>()
            + tree.edges.allocation_size()
            + tree.roots.allocation_size();
        assert!(tree.edges.allocation_size() > 0 && tree.roots.allocation_size() > 0);
        assert_eq!(tree.bytes(), tree.content_bytes + containers);
        for node in tree.nodes.iter().flatten() {
            // At least the struct, the edge text, the IDs and the `Arc` counts.
            let raw = size_of::<Node>() + node.text.len() + 4 * node.ids.len() + 16;
            assert!(node.bytes() >= raw);
        }
    }

    #[test]
    fn evictions_release_container_capacity() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(L1CacheConfig::default());
        for i in 0..256 {
            check(&encoder, &tokenizer, &format!("{i}</s>"));
        }
        let (edges, bytes) = {
            let tree = cache.tree.read();
            (tree.edges.capacity(), tree.bytes())
        };
        assert!(edges >= 256);
        {
            let mut tree = cache.tree.write();
            // Keep the root and one leaf.
            assert_eq!(tree.evict(usize::MAX, 2), 255);
            assert!(tree.edges.capacity() < edges / 4);
            assert!(tree.bytes() < bytes / 4);
        }
        check_invariants(&cache);
        // An empty tree holds no storage at all.
        cache.tree.write().evict(0, 0);
        assert_eq!(cache.stats().bytes, 0);
        assert!(check(&encoder, &tokenizer, "0</s>") == 0);
        check_invariants(&cache);
    }

    #[test]
    fn eviction_is_least_recently_used() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(L1CacheConfig {
            max_nodes: 4,
            ..Default::default()
        });
        for text in ["a</s>", "b</s>", "c</s>"] {
            check(&encoder, &tokenizer, text);
        }
        // A hit refreshes `a`, so `b` is the oldest leaf when `d` arrives.
        assert!(check(&encoder, &tokenizer, "a</s>") > 0);
        check(&encoder, &tokenizer, "d</s>");
        assert_eq!(cache.stats().evictions, 1);
        for text in ["a</s>", "c</s>", "d</s>"] {
            assert!(check(&encoder, &tokenizer, text) > 0, "{text}");
        }
        assert_eq!(check(&encoder, &tokenizer, "b</s>"), 0);
        check_invariants(&cache);
    }

    #[test]
    fn insert_walks_again_when_the_tree_changed_after_lookup() {
        let tokenizer = fixture_tokenizer();
        let (encoder, cache) = encoder(Default::default());
        let namespace = Namespace {
            instance: 0,
            salt: None,
        };
        let insert_after = |text: &str, change: &dyn Fn()| {
            let first_cut = |from| encoder.matcher.first_cut(text, from);
            let hit = cache.lookup(&namespace, text, first_cut);
            change();
            let cuts = encoder.matcher.cuts(text, hit.bytes);
            let tail = tokenizer.encode(&text[hit.bytes..], false).unwrap();
            let bounds = encoder
                .tail_bounds(&tokenizer, text, &hit, &cuts, &tail)
                .unwrap();
            let mut ids: Vec<u32> = hit
                .pieces
                .iter()
                .flat_map(|(piece, take)| piece[..*take].to_vec())
                .collect();
            ids.extend(tail);
            cache.insert(&namespace, text, &ids, hit, &bounds, first_cut);
        };
        // Another request stores a shared prefix in between: the insert
        // walks again, splits that edge and appends only its own turn.
        let text = "<s>sys</s><s>user one</s><s>";
        insert_after(text, &|| {
            check(&encoder, &tokenizer, "<s>sys</s><s>user two</s><s>");
        });
        check_invariants(&cache);
        let all = tokenizer.encode(text, false).unwrap().len();
        assert_eq!(check(&encoder, &tokenizer, text), all);
        // An eviction removes the matched path in between: the cuts before
        // the match have no offsets, so nothing is stored.
        insert_after(&format!("{text}assistant</s>"), &|| {
            cache.tree.write().evict(0, 0);
        });
        check_invariants(&cache);
        assert_eq!(cache.stats().nodes, 0);
    }

    #[test]
    fn cut_offsets_fall_back_to_segments_when_boundary_ids_do_not_line_up() {
        use std::sync::atomic::AtomicUsize;

        // Emits `<s>`'s ID for plain "zz" too, which never crosses a cut.
        struct Spurious {
            inner: HuggingFaceTokenizer,
            calls: AtomicUsize,
        }
        impl Tokenizer for Spurious {
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.inner.encode(&text.replace("zz", "<s>"), special)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.inner.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.inner.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.inner.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.inner.id_to_token(id)
            }
        }
        let tokenizer = Spurious {
            inner: fixture_tokenizer(),
            calls: AtomicUsize::new(0),
        };
        let (encoder, cache) = encoder(Default::default());
        let text = "<s>zz one</s><s>two</s><s>";
        assert_eq!(check(&encoder, &tokenizer, text), 0);
        // The tail encode, one encode per segment for the offsets, and the
        // reference encode in `check`.
        assert_eq!(tokenizer.calls.load(Ordering::Relaxed), 1 + 5 + 1);
        let all = tokenizer.encode(text, false).unwrap().len();
        assert_eq!(check(&encoder, &tokenizer, text), all);
        check_invariants(&cache);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn salt_is_one_allocation_and_the_insert_cap_counts_marks() {
        let tokenizer = fixture_tokenizer();
        let text = "<s>system hi</s><s>user</s>";
        {
            let (encoder, cache) = encoder(Default::default());
            check(&encoder, &tokenizer, text);
            let plain = cache.tree.read().content_bytes;
            let salt = "abcdef";
            // Same text again, so this lookup hits.
            check(&encoder, &tokenizer, text);
            // A new salt is a new root and leaf.
            let (_, _) = encoder
                .encode(&tokenizer, 0, text, Some(salt))
                .unwrap()
                .unwrap();
            assert_eq!(
                cache.tree.read().content_bytes,
                plain * 2 + heap_bytes(2 * size_of::<usize>() + salt.len())
            );
            {
                let tree = cache.tree.read();
                let root = tree
                    .nodes
                    .iter()
                    .flatten()
                    .find(|node| {
                        node.namespace
                            .as_ref()
                            .and_then(|ns| ns.salt.as_ref())
                            .is_some_and(|stored| stored.as_ref() == salt)
                    })
                    .unwrap();
                // The node and the roots map share one string.
                assert_eq!(
                    Arc::strong_count(root.namespace.as_ref().unwrap().salt.as_ref().unwrap()),
                    2
                );
            }
            check_invariants(&cache);
        }

        // Nine `<s>` tokens: the leaf and the new namespace root fit the
        // insert cap, but not once the leaf's interior marks are counted.
        let nine = "<s>".repeat(9);
        let ids = tokenizer.encode(&nine, false).unwrap();
        let leaf_alone = node_bytes(nine.len(), ids.len(), 0, None) + node_bytes(0, 0, 0, None);
        let (encoder, cache) = encoder(L1CacheConfig {
            max_insert_bytes: leaf_alone,
            max_nodes: 32,
            ..Default::default()
        });
        check(&encoder, &tokenizer, &nine);
        let tree = cache.tree.read();
        assert!(
            tree.nodes
                .iter()
                .flatten()
                .all(|node| node.text.len() != nine.len()),
            "the full leaf is over the insert cap once marks and the root are counted"
        );
        assert!(tree.live > 0 && tree.content_bytes <= leaf_alone);
        drop(tree);
        check_invariants(&cache);
    }

    #[test]
    fn eviction_keeps_budget_and_recent_paths() {
        let tokenizer = fixture_tokenizer();
        let config = L1CacheConfig {
            max_bytes: 8 * 1024,
            max_nodes: 16,
            max_insert_bytes: 1024,
            max_insert_segments: 8,
            ..Default::default()
        };
        let (encoder, cache) = encoder(config);
        for i in 0..200 {
            check(
                &encoder,
                &tokenizer,
                &format!("<s>system {}</s><s>user {i}</s><s>a", i % 7),
            );
            let stats = cache.stats();
            assert!(stats.bytes <= config.max_bytes && stats.nodes <= config.max_nodes);
        }
        assert!(cache.stats().evictions > 0);
        let recent = "<s>system 3</s><s>user 199</s><s>a";
        assert!(check(&encoder, &tokenizer, recent) > 0);
        // Long segments past max_insert_bytes are encoded but not stored.
        let before = cache.stats();
        check(
            &encoder,
            &tokenizer,
            &format!("<s>{}</s>", "long ".repeat(400)),
        );
        assert_eq!(cache.stats().oversized, before.oversized + 1);
        assert!(cache.stats().bytes <= config.max_bytes);
        // Segment cap stores only the first segments.
        let many: String = (0..20).map(|i| format!("<s>m{i}</s>")).collect();
        check(&encoder, &tokenizer, &many);
        assert!(check(&encoder, &tokenizer, &many) > 0);
    }

    #[test]
    fn concurrent_encodes_match_uncached() {
        let tokenizer = Arc::new(fixture_tokenizer());
        let (encoder, cache) = encoder(L1CacheConfig {
            max_bytes: 64 * 1024,
            max_nodes: 64,
            max_insert_bytes: 8 * 1024,
            ..Default::default()
        });
        std::thread::scope(|scope| {
            for t in 0..8 {
                let (encoder, tokenizer) = (encoder.clone(), tokenizer.clone());
                scope.spawn(move || {
                    for i in 0..200 {
                        let turns: String = (0..(i % 9))
                            .map(|k| format!("<s>user {}</s><s>assistant {k}</s>", (k + t) % 3))
                            .collect();
                        check(
                            &encoder,
                            tokenizer.as_ref(),
                            &format!("<s>sys</s>{turns}<s>"),
                        );
                    }
                });
            }
        });
        let stats = cache.stats();
        assert_eq!(stats.lookups, 1600);
        assert!(stats.hits > 0 && stats.nodes <= 64);
    }

    struct RecordingGauge {
        name: &'static str,
        sets: Arc<Mutex<Vec<&'static str>>>,
    }

    impl GaugeFn for RecordingGauge {
        fn increment(&self, _: f64) {}
        fn decrement(&self, _: f64) {}
        fn set(&self, _: f64) {
            self.sets.lock().push(self.name);
        }
    }

    struct PairRecorder(Arc<Mutex<Vec<&'static str>>>);

    impl Recorder for PairRecorder {
        fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
        fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}

        fn register_counter(&self, _: &Key, _: &Metadata<'_>) -> Counter {
            Counter::noop()
        }

        fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
            let name = if key.name() == "vllm_tokenizer_l1_nodes" {
                "nodes"
            } else {
                "bytes"
            };
            Gauge::from_arc(Arc::new(RecordingGauge {
                name,
                sets: self.0.clone(),
            }))
        }

        fn register_histogram(&self, _: &Key, _: &Metadata<'_>) -> Histogram {
            Histogram::noop()
        }
    }

    /// Two caches publish both gauges under one lock, so their `set` calls
    /// cannot interleave.
    #[test]
    fn occupancy_publishes_nodes_and_bytes_together() {
        let sets = Arc::new(Mutex::new(Vec::new()));
        let tokenizer = Arc::new(fixture_tokenizer());
        thread::scope(|scope| {
            for t in 0..4 {
                let (sets, tokenizer) = (sets.clone(), tokenizer.clone());
                scope.spawn(move || {
                    metrics::with_local_recorder(&PairRecorder(sets), || {
                        let (encoder, _cache) = encoder(L1CacheConfig {
                            max_bytes: 64 * 1024,
                            max_nodes: 64,
                            max_insert_bytes: 8 * 1024,
                            ..Default::default()
                        });
                        check(
                            &encoder,
                            tokenizer.as_ref(),
                            &format!("<s>user {t}</s><s>tail {t}</s>"),
                        );
                    });
                });
            }
        });
        let sets = sets.lock();
        assert!(sets.len() >= 8, "{sets:?}");
        for pair in sets.chunks(2) {
            assert_eq!(pair, ["nodes", "bytes"], "{sets:?}");
        }
    }

    fn lookup_report_node(parent: usize, text: String) -> Node {
        Node::new(
            parent,
            Box::from(text),
            Arc::from([]),
            Box::from([]),
            0,
            None,
        )
    }

    /// Children of `parent` whose first segment shares `segment`'s key.
    fn bucket_len(tree: &Tree, parent: usize, segment: &str) -> usize {
        let key = EdgeKey {
            parent,
            segment: tree.key(segment),
        };
        let mut id = tree.edges.get(&key).copied().unwrap_or(NO_NODE);
        let mut len = 0;
        while id != NO_NODE {
            len += 1;
            id = tree.node(id).next_same_key;
        }
        len
    }

    fn timed_lookup(
        iters: usize,
        mut lookup: impl FnMut() -> Option<usize>,
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        let mut hits = 0usize;
        for _ in 0..iters {
            if std::hint::black_box(lookup()).is_some() {
                hits += 1;
            }
        }
        let elapsed = start.elapsed();
        assert_eq!(hits, iters);
        elapsed
    }

    fn timed_encode(
        iters: usize,
        encoder: &L1Encoder,
        tokenizer: &dyn Tokenizer,
        text: &str,
    ) -> std::time::Duration {
        let start = std::time::Instant::now();
        let mut total = 0usize;
        for _ in 0..iters {
            let (ids, cached) = encoder
                .encode(tokenizer, 0, text, None)
                .unwrap()
                .expect("text has boundaries");
            total = total.wrapping_add(ids.len()).wrapping_add(cached);
        }
        let elapsed = start.elapsed();
        std::hint::black_box(total);
        elapsed
    }

    fn normal_long_segment(i: usize, len: usize) -> String {
        let head = "h".repeat(64);
        let tail = format!("{i:064x}");
        let middle = format!("normal-{i}-").repeat((len - 128) / 9 + 1);
        format!("{head}{}{tail}", &middle[..len - 128])
    }

    fn adversarial_long_segment(i: usize, len: usize) -> String {
        let head = "h".repeat(64);
        let tail = "t".repeat(64);
        let marker = format!("{i:04x}");
        let middle = format!("{}{}", "m".repeat(len - 128 - marker.len()), marker);
        format!("{head}{middle}{tail}")
    }

    fn build_lookup_tree(segments: &[String], strategy: L1HashStrategy) -> (Tree, usize) {
        let mut tree = Tree::new(strategy);
        let root = tree.alloc(lookup_report_node(NO_NODE, String::new()));
        for segment in segments {
            let id = tree.alloc(lookup_report_node(root, segment.clone()));
            tree.add_child(root, id);
        }
        (tree, root)
    }

    fn hash_report_encoder(strategy: L1HashStrategy) -> (L1Encoder, Arc<L1Cache>) {
        encoder(L1CacheConfig {
            max_bytes: 512 * 1024 * 1024,
            max_nodes: 1_000_000,
            max_insert_bytes: 256 * 1024 * 1024,
            max_insert_segments: 1_000_000,
            hash_strategy: strategy,
            ..Default::default()
        })
    }

    fn hash_report_prompt(segment: &str) -> String {
        format!("<s>{segment}</s>")
    }

    fn warm_hash_report_encoder(
        encoder: &L1Encoder,
        tokenizer: &dyn Tokenizer,
        segments: &[String],
    ) {
        for segment in segments {
            check(encoder, tokenizer, &hash_report_prompt(segment));
        }
    }

    fn strategy_name(strategy: L1HashStrategy) -> &'static str {
        match strategy {
            L1HashStrategy::FirstLast64 => "first_last_64",
            L1HashStrategy::Full => "full",
        }
    }

    fn prompt_tokens(tokenizer: &dyn Tokenizer, segment: &str) -> usize {
        tokenizer
            .encode(&hash_report_prompt(segment), false)
            .unwrap()
            .len()
    }

    fn byte_len_for_prompt_tokens(tokenizer: &dyn Tokenizer, target_tokens: usize) -> usize {
        let mut high = 129usize;
        while prompt_tokens(tokenizer, &normal_long_segment(0, high)) < target_tokens {
            high *= 2;
        }
        let mut low = 129usize;
        while low < high {
            let mid = (low + high) / 2;
            if prompt_tokens(tokenizer, &normal_long_segment(0, mid)) < target_tokens {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }

    /// CPU-only lookup and encode report for the current first/last-64 child
    /// key versus the full-segment hash strategy.
    ///
    /// Run with:
    /// `cargo test --release --lib hash_strategy_lookup_cost_report -- --ignored --nocapture`
    #[test]
    #[ignore = "CPU-only performance report; run in release mode"]
    fn hash_strategy_lookup_cost_report() {
        fn report_lookup_row(
            case: &str,
            len: usize,
            siblings: usize,
            first_last_bucket: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_ns = first_last.as_nanos() as f64 / iters as f64;
            let full_hash_ns = full_hash.as_nanos() as f64 / iters as f64;
            println!(
                "lookup {case:11} len={len:6} siblings={siblings:2} first_last_bucket={first_last_bucket:2} iters={iters:5} first_last_64={first_last_ns:9.1}ns full={full_hash_ns:9.1}ns ratio={:.2}",
                first_last_ns / full_hash_ns
            );
        }

        fn report_encode_row(
            case: &str,
            len: usize,
            siblings: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_us = first_last.as_secs_f64() * 1_000_000.0 / iters as f64;
            let full_hash_us = full_hash.as_secs_f64() * 1_000_000.0 / iters as f64;
            println!(
                "encode {case:11} len={len:6} siblings={siblings:2} iters={iters:5} first_last_64={first_last_us:9.2}us full={full_hash_us:9.2}us ratio={:.2}",
                first_last_us / full_hash_us
            );
        }

        println!(
            "hash strategy matrix; lookup rows include the exact `first_segment() == segment` check"
        );
        println!(
            "normal: siblings share the first 64 bytes but have different last 64 bytes, so current bucket is 1"
        );
        println!(
            "adversarial: siblings share length, first 64 bytes and last 64 bytes, so current bucket equals siblings"
        );

        let tokenizer = fixture_tokenizer();
        for len in [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024] {
            for siblings in [1, 4, 8, 16, 32] {
                for (case, segments) in [
                    (
                        "normal",
                        (0..siblings)
                            .map(|i| normal_long_segment(i, len))
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "adversarial",
                        (0..siblings)
                            .map(|i| adversarial_long_segment(i, len))
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let query = &segments[siblings - 1];
                    assert!(query.len() > 128);

                    let (first_last_tree, first_last_root) =
                        build_lookup_tree(&segments, L1HashStrategy::FirstLast64);
                    let (full_tree, full_root) = build_lookup_tree(&segments, L1HashStrategy::Full);
                    let first_last_bucket = bucket_len(&first_last_tree, first_last_root, query);
                    assert_eq!(
                        first_last_bucket,
                        if case == "normal" { 1 } else { siblings }
                    );
                    let lookup_iters = if case == "normal" {
                        (64 * 1024 * 1024 / len).max(512)
                    } else {
                        (64 * 1024 * 1024 / (len * siblings)).max(32)
                    };
                    let first_last_lookup = timed_lookup(lookup_iters, || {
                        first_last_tree.child(first_last_root, query)
                    });
                    let full_lookup =
                        timed_lookup(lookup_iters, || full_tree.child(full_root, query));
                    report_lookup_row(
                        case,
                        len,
                        siblings,
                        first_last_bucket,
                        lookup_iters,
                        first_last_lookup,
                        full_lookup,
                    );

                    let (first_last_encoder, _) = hash_report_encoder(L1HashStrategy::FirstLast64);
                    let (full_encoder, _) = hash_report_encoder(L1HashStrategy::Full);
                    warm_hash_report_encoder(&first_last_encoder, &tokenizer, &segments);
                    warm_hash_report_encoder(&full_encoder, &tokenizer, &segments);
                    let query_text = hash_report_prompt(query);
                    let (ids, cached) = first_last_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let (ids, cached) = full_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let encode_iters = (8 * 1024 * 1024 / len).max(8);
                    let first_last_encode =
                        timed_encode(encode_iters, &first_last_encoder, &tokenizer, &query_text);
                    let full_encode =
                        timed_encode(encode_iters, &full_encoder, &tokenizer, &query_text);
                    report_encode_row(
                        case,
                        len,
                        siblings,
                        encode_iters,
                        first_last_encode,
                        full_encode,
                    );
                }
            }
        }

        println!(
            "Set VLLM_ROUTER_L1_HASH_STRATEGY={} or {} to switch the runtime strategy.",
            strategy_name(L1HashStrategy::FirstLast64),
            strategy_name(L1HashStrategy::Full)
        );
    }

    /// Same report as `hash_strategy_lookup_cost_report`, but targets prompt
    /// token counts rather than raw segment byte lengths.
    ///
    /// Run with:
    /// `cargo test --release --lib hash_strategy_token_count_report -- --ignored --nocapture`
    #[test]
    #[ignore = "CPU-only performance report; run in release mode"]
    fn hash_strategy_token_count_report() {
        #[derive(Clone, Copy)]
        struct ReportDims {
            case: &'static str,
            target_tokens: usize,
            actual_tokens: usize,
            bytes: usize,
            siblings: usize,
        }

        fn report_lookup_row(
            dims: ReportDims,
            first_last_bucket: usize,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_ns = first_last.as_nanos() as f64 / iters as f64;
            let full_hash_ns = full_hash.as_nanos() as f64 / iters as f64;
            let ReportDims {
                case,
                target_tokens,
                actual_tokens,
                bytes,
                siblings,
            } = dims;
            println!(
                "lookup {case:11} target_tokens={target_tokens:6} actual_tokens={actual_tokens:6} bytes={bytes:7} siblings={siblings:2} first_last_bucket={first_last_bucket:2} iters={iters:4} first_last_64={first_last_ns:10.1}ns full={full_hash_ns:10.1}ns ratio={:.2}",
                first_last_ns / full_hash_ns
            );
        }

        fn report_encode_row(
            dims: ReportDims,
            iters: usize,
            first_last: std::time::Duration,
            full_hash: std::time::Duration,
        ) {
            let first_last_us = first_last.as_secs_f64() * 1_000_000.0 / iters as f64;
            let full_hash_us = full_hash.as_secs_f64() * 1_000_000.0 / iters as f64;
            let ReportDims {
                case,
                target_tokens,
                actual_tokens,
                bytes,
                siblings,
            } = dims;
            println!(
                "encode {case:11} target_tokens={target_tokens:6} actual_tokens={actual_tokens:6} bytes={bytes:7} siblings={siblings:2} iters={iters:4} first_last_64={first_last_us:10.2}us full={full_hash_us:10.2}us ratio={:.2}",
                first_last_us / full_hash_us
            );
        }

        println!("hash strategy token-count matrix; target length is prompt token count, not KiB");
        let tokenizer = fixture_tokenizer();
        for target_tokens in [16_384, 32_768, 65_536, 131_072] {
            let bytes = byte_len_for_prompt_tokens(&tokenizer, target_tokens);
            for siblings in [1, 4, 8, 16, 32] {
                for (case, segments) in [
                    (
                        "normal",
                        (0..siblings)
                            .map(|i| normal_long_segment(i, bytes))
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "adversarial",
                        (0..siblings)
                            .map(|i| adversarial_long_segment(i, bytes))
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let query = &segments[siblings - 1];
                    let query_text = hash_report_prompt(query);
                    let actual_tokens = tokenizer.encode(&query_text, false).unwrap().len();
                    let dims = ReportDims {
                        case,
                        target_tokens,
                        actual_tokens,
                        bytes,
                        siblings,
                    };
                    let (first_last_tree, first_last_root) =
                        build_lookup_tree(&segments, L1HashStrategy::FirstLast64);
                    let (full_tree, full_root) = build_lookup_tree(&segments, L1HashStrategy::Full);
                    let first_last_bucket = bucket_len(&first_last_tree, first_last_root, query);
                    assert_eq!(
                        first_last_bucket,
                        if case == "normal" { 1 } else { siblings }
                    );
                    let lookup_iters = if case == "normal" {
                        (64 * 1024 * 1024 / bytes).max(64)
                    } else {
                        (64 * 1024 * 1024 / (bytes * siblings)).max(16)
                    };
                    let first_last_lookup = timed_lookup(lookup_iters, || {
                        first_last_tree.child(first_last_root, query)
                    });
                    let full_lookup =
                        timed_lookup(lookup_iters, || full_tree.child(full_root, query));
                    report_lookup_row(
                        dims,
                        first_last_bucket,
                        lookup_iters,
                        first_last_lookup,
                        full_lookup,
                    );

                    let (first_last_encoder, _) = hash_report_encoder(L1HashStrategy::FirstLast64);
                    let (full_encoder, _) = hash_report_encoder(L1HashStrategy::Full);
                    warm_hash_report_encoder(&first_last_encoder, &tokenizer, &segments);
                    warm_hash_report_encoder(&full_encoder, &tokenizer, &segments);
                    let (ids, cached) = first_last_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let (ids, cached) = full_encoder
                        .encode(&tokenizer, 0, &query_text, None)
                        .unwrap()
                        .expect("query has boundaries");
                    assert_eq!(ids, tokenizer.encode(&query_text, false).unwrap());
                    assert_eq!(cached, ids.len());
                    let encode_iters = (1_000_000 / target_tokens).max(4);
                    let first_last_encode =
                        timed_encode(encode_iters, &first_last_encoder, &tokenizer, &query_text);
                    let full_encode =
                        timed_encode(encode_iters, &full_encoder, &tokenizer, &query_text);
                    report_encode_row(dims, encode_iters, first_last_encode, full_encode);
                }
            }
        }
    }

    fn check_invariants(cache: &L1Cache) {
        let tree = cache.tree.read();
        let live: Vec<usize> = (0..tree.nodes.len())
            .filter(|&id| tree.nodes[id].is_some())
            .collect();
        assert_eq!(live.len(), tree.live);
        assert_eq!(
            live.iter().map(|&id| tree.node(id).bytes()).sum::<usize>(),
            tree.content_bytes
        );
        assert!(tree.bytes() <= cache.config.max_bytes);
        assert!(tree.live <= cache.config.max_nodes);
        // Each edge entry chains children of its parent under its key.
        let mut children = vec![0; tree.nodes.len()];
        for (key, &head) in &tree.edges {
            let mut id = head;
            while id != NO_NODE {
                assert_eq!(tree.node(id).parent, key.parent);
                assert!(tree.edge(key.parent, id) == *key);
                children[key.parent] += 1;
                id = tree.node(id).next_same_key;
            }
        }
        assert_eq!(
            children.iter().sum::<usize>(),
            live.len() - tree.roots.len()
        );
        let queued: HashSet<usize> = tree.leaves.iter().map(|&Reverse((_, id))| id).collect();
        for &id in &live {
            let node = tree.node(id);
            assert_eq!(children[id], node.child_count);
            if let Some(namespace) = &node.namespace {
                assert_eq!(tree.roots[namespace], id);
                assert!(node.child_count > 0, "empty roots are removed");
                continue;
            }
            assert!(!node.text.is_empty());
            if node.is_leaf() {
                assert!(queued.contains(&id), "every leaf has a heap entry");
            }
            let mut prev = (0, 0);
            for &(byte, token) in node.marks.iter() {
                assert!(byte > prev.0 && token > prev.1);
                assert!((byte as usize) < node.text.len() && (token as usize) <= node.ids.len());
                prev = (byte, token);
            }
        }
    }

    #[test]
    fn randomized_conversations_match_uncached_and_keep_invariants() {
        let tokenizer = fixture_tokenizer();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        };
        let words = [
            "hi", " there", "\n", "数", "😀", "<s", "s>", "</", " \t", "1", "é",
        ];
        for budget in [8 * 1024, 16 * 1024, 1 << 20] {
            let (encoder, cache) = encoder(L1CacheConfig {
                max_bytes: budget,
                max_nodes: 24,
                // The insert cap is the whole budget, so inserts also evict.
                max_insert_bytes: budget,
                max_insert_segments: 6,
                ..Default::default()
            });
            let mut conversations: Vec<Vec<String>> = vec![Vec::new(); 4];
            for _ in 0..400 {
                let conversation = &mut conversations[next(4)];
                match next(4) {
                    0 if !conversation.is_empty() => {
                        let at = next(conversation.len());
                        conversation.truncate(at);
                    }
                    1 if !conversation.is_empty() => {
                        let at = next(conversation.len());
                        conversation[at].push_str(words[next(words.len())]);
                    }
                    _ => {
                        conversation.push((0..next(4)).map(|_| words[next(words.len())]).collect())
                    }
                }
                let mut text: String = conversation
                    .iter()
                    .map(|message| format!("<s>{message}</s>"))
                    .collect();
                text.push_str(["", "<s>", "<s>assistant", "tail"][next(4)]);
                if let Some((ids, _)) = encoder.encode(&tokenizer, 0, &text, None).unwrap() {
                    assert_eq!(ids, tokenizer.encode(&text, false).unwrap(), "{text:?}");
                }
                check_invariants(&cache);
                let stats = cache.stats();
                assert!(stats.bytes <= budget && stats.nodes <= 24);
            }
            assert!(cache.stats().hits > 0);
        }
    }

    /// Prints every added token with its L1 verdict for the models in
    /// `VLLM_ROUTER_L1_TOKENIZERS` (local dirs or hub IDs).
    #[tokio::test]
    async fn report_boundary_verdicts() {
        let Ok(models) = std::env::var("VLLM_ROUTER_L1_TOKENIZERS") else {
            eprintln!("skip: set VLLM_ROUTER_L1_TOKENIZERS");
            return;
        };
        for model in models.split(',').filter(|m| !m.is_empty()) {
            let added = match load_added_tokens(model).await {
                Ok(added) => added,
                Err(error) => {
                    eprintln!("== {model}: L1 unavailable: {error:#}");
                    continue;
                }
            };
            let files = ResolvedModelFiles::new(model).await.unwrap();
            let TokenizerSource::HuggingFace(path) = files.tokenizer else {
                unreachable!()
            };
            let tokenizer = HuggingFaceTokenizer::new(&path).unwrap();
            let probed = BoundaryMatcher::new(&added)
                .and_then(|matcher| matcher.probed(&tokenizer))
                .map(|matcher| matcher.tokens().to_vec())
                .unwrap_or_default();
            eprintln!(
                "== {model}: {} added tokens, {} boundaries",
                added.tokens.len(),
                probed.len()
            );
            for (token, rejection) in added.tokens.iter().zip(boundary_rejections(&added)) {
                let verdict = match rejection {
                    Some(reason) => format!("no   {reason}"),
                    None if probed.contains(&token.content) => "YES".into(),
                    None => "no   failed load-time probe".into(),
                };
                eprintln!("  {verdict:<55} {:?}", token.content);
            }
        }
    }

    /// Real tokenizers: `VLLM_ROUTER_L1_TOKENIZERS=a,b` (local dirs or hub
    /// IDs). Random prompts mix boundary tokens, all other added tokens,
    /// fragments of added tokens and adversarial text.
    #[tokio::test]
    async fn randomized_prompts_match_real_tokenizers() {
        let Ok(dirs) = std::env::var("VLLM_ROUTER_L1_TOKENIZERS") else {
            eprintln!("skip: set VLLM_ROUTER_L1_TOKENIZERS");
            return;
        };
        for dir in dirs.split(',').filter(|d| !d.is_empty()) {
            let added = match load_added_tokens(dir).await {
                Ok(added) => added,
                Err(error) => {
                    eprintln!("{dir}: L1 unavailable: {error}");
                    continue;
                }
            };
            let files = ResolvedModelFiles::new(dir).await.unwrap();
            let TokenizerSource::HuggingFace(path) = files.tokenizer else {
                unreachable!()
            };
            let tokenizer = HuggingFaceTokenizer::new(&path).unwrap();
            let matcher = match BoundaryMatcher::new(&added) {
                Ok(matcher) => matcher.probed(&tokenizer).unwrap(),
                Err(error) => {
                    eprintln!("{dir}: L1 unavailable: {error}");
                    continue;
                }
            };
            let added: Vec<String> = added
                .tokens
                .into_iter()
                .map(|token| token.content)
                .filter(|content| !content.is_empty())
                .collect();
            let boundaries = matcher.tokens().to_vec();
            let cache = Arc::new(
                L1Cache::new(L1CacheConfig {
                    max_bytes: 1 << 20,
                    max_nodes: 512,
                    max_insert_bytes: 1 << 18,
                    ..Default::default()
                })
                .unwrap(),
            );
            let encoder = L1Encoder::new(dir, cache.clone(), matcher);
            let words = [
                "hello", " world", "  ", "\n", "\n\n", "\t", "数据", "😀", "é", "\u{301}",
                "\u{338}", ">", "<", "|", "▁", "Ġ", "'s", "123", " 4", "<|", "|>", "</", "\u{a0}",
                "ﬁ", "Å",
            ];
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut next = move |n: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % n as u64) as usize
            };
            let mut history: Vec<String> = Vec::new();
            for _ in 0..3000 {
                let mut text = match next(3) {
                    0 if !history.is_empty() => history[next(history.len())].clone(),
                    _ => String::new(),
                };
                for _ in 0..1 + next(12) {
                    match next(5) {
                        0 => text.push_str(&boundaries[next(boundaries.len())]),
                        1 => text.push_str(&added[next(added.len())]),
                        2 => {
                            // A prefix or suffix of an added token, so that
                            // longer tokens can form across neighbours.
                            let token = &added[next(added.len())];
                            let cuts: Vec<usize> = token.char_indices().map(|(i, _)| i).collect();
                            let at = cuts[next(cuts.len())];
                            text.push_str(if next(2) == 0 {
                                &token[..at]
                            } else {
                                &token[at..]
                            });
                        }
                        _ => text.push_str(words[next(words.len())]),
                    }
                }
                let full = tokenizer.encode(&text, false).unwrap();
                if let Some((ids, _)) = encoder.encode(&tokenizer, 0, &text, None).unwrap() {
                    assert_eq!(ids, full, "{dir}: {text:?}");
                }
                history.push(text);
            }
            let stats = cache.stats();
            eprintln!("{dir}: {} boundaries, {stats:?}", boundaries.len());
            assert!(stats.hits > 0);
        }
    }

    #[test]
    fn tail_beyond_the_insert_cap_is_one_encode() {
        use std::sync::atomic::AtomicUsize;

        struct Counting {
            inner: HuggingFaceTokenizer,
            calls: AtomicUsize,
        }
        impl Tokenizer for Counting {
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.inner.encode(text, special)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.inner.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.inner.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.inner.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.inner.id_to_token(id)
            }
        }

        let tokenizer = Counting {
            inner: fixture_tokenizer(),
            calls: AtomicUsize::new(0),
        };
        let (encoder, _) = encoder(L1CacheConfig {
            max_insert_segments: 2,
            ..Default::default()
        });
        let text = "<s>one</s><s>two</s><s>three</s><s>four</s>tail";
        let (ids, cached) = encoder.encode(&tokenizer, 0, text, None).unwrap().unwrap();
        assert_eq!(ids, tokenizer.inner.encode(text, false).unwrap());
        assert_eq!(cached, 0);
        // One encode of the tail; boundary IDs give the stored offsets.
        assert_eq!(tokenizer.calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn probe_drops_boundaries_that_change_encoding() {
        struct Merging(HuggingFaceTokenizer);
        impl Tokenizer for Merging {
            // Pretend `<s>` merges with a following "hi".
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode(&text.replace("<s>hi", "<s>"), special)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.0.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.0.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
        }
        let set = token_set(vec![spec("<s>"), spec("</s>")], true);
        let probed = BoundaryMatcher::new(&set)
            .unwrap()
            .probed(&Merging(fixture_tokenizer()))
            .unwrap();
        assert_eq!(probed.tokens(), ["</s>"]);
        let all = BoundaryMatcher::new(&set).unwrap();
        assert_eq!(all.probed(&fixture_tokenizer()).unwrap().tokens().len(), 2);
    }

    #[test]
    fn verify_mode_disables_l1_on_mismatch() {
        struct Drifting(HuggingFaceTokenizer);
        impl Tokenizer for Drifting {
            fn encode(&self, text: &str, special: bool) -> vllm_tokenizer::Result<Vec<u32>> {
                let mut ids = self.0.encode(text, special)?;
                if text.len() > 20 {
                    ids.push(1);
                }
                Ok(ids)
            }
            fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
                self.0.encode_ordinary(text)
            }
            fn decode(&self, ids: &[u32], special: bool) -> vllm_tokenizer::Result<String> {
                self.0.decode(ids, special)
            }
            fn token_to_id(&self, token: &str) -> Option<u32> {
                self.0.token_to_id(token)
            }
            fn id_to_token(&self, id: u32) -> Option<String> {
                self.0.id_to_token(id)
            }
        }
        let tokenizer = Drifting(fixture_tokenizer());
        let (encoder, cache) = encoder(L1CacheConfig {
            verify: true,
            ..Default::default()
        });
        let text = "<s>system hello there</s><s>user hi</s>";
        let expected = (tokenizer.encode(text, false).unwrap(), 0);
        // The cold request is one encode of the whole prompt; the reuse on
        // the next request is what disagrees.
        let encode = || encoder.encode(&tokenizer, 0, text, None).unwrap();
        assert_eq!(encode().unwrap(), expected);
        assert_eq!(cache.stats().verify_mismatches, 0);
        assert_eq!(encode().unwrap(), expected);
        assert_eq!(cache.stats().verify_mismatches, 1);
        assert!(encoder.encode(&tokenizer, 0, text, None).unwrap().is_none());
        assert_eq!(cache.stats().bypasses, 1);
    }

    #[test]
    fn bypasses_are_counted_only_when_l1_is_skipped() {
        use super::super::encode_cache::PromptEncoder;
        use vllm_text::Prompt;

        let (l1, cache) = encoder(Default::default());
        let prompts = PromptEncoder::new("fixture", Arc::new(fixture_tokenizer()), None)
            .with_l1_cache(Some(l1));
        let encode = |prompt, add_special_tokens| {
            prompts.encode(prompt, add_special_tokens, None).unwrap();
        };
        encode(Prompt::TokenIds(vec![1, 2]), false);
        encode(Prompt::Text("<s>hi</s>".into()), true);
        encode(Prompt::Text("no boundary here".into()), false);
        assert_eq!((cache.stats().bypasses, cache.stats().lookups), (3, 0));
        encode(Prompt::Text("<s>hi</s>".into()), false);
        assert_eq!((cache.stats().bypasses, cache.stats().lookups), (3, 1));
    }

    #[test]
    fn config_is_opt_in_and_validated() {
        let parse = |values: &[(&str, &str)]| {
            parse_config(|key| {
                values
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(parse(&[("VLLM_ROUTER_L1_MAX_BYTES", "bad")]).unwrap(), None);
        assert_eq!(
            parse(&[("VLLM_ROUTER_L1_CACHE", "1")]).unwrap(),
            Some(Default::default())
        );
        let small = parse(&[
            ("VLLM_ROUTER_L1_CACHE", "true"),
            ("VLLM_ROUTER_L1_MAX_BYTES", "1000"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!((small.max_bytes, small.max_insert_bytes), (1000, 1000));
        assert!(
            parse(&[
                ("VLLM_ROUTER_L1_CACHE", "1"),
                ("VLLM_ROUTER_L1_VERIFY", "1")
            ])
            .unwrap()
            .unwrap()
            .verify
        );
        assert_eq!(
            parse(&[
                ("VLLM_ROUTER_L1_CACHE", "1"),
                ("VLLM_ROUTER_L1_HASH_STRATEGY", "full")
            ])
            .unwrap()
            .unwrap()
            .hash_strategy,
            L1HashStrategy::Full
        );
        for (name, value) in [
            ("VLLM_ROUTER_L1_CACHE", "maybe"),
            ("VLLM_ROUTER_L1_MAX_NODES", "0"),
            ("VLLM_ROUTER_L1_MAX_INSERT_SEGMENTS", "x"),
            ("VLLM_ROUTER_L1_VERIFY", "yes"),
            ("VLLM_ROUTER_L1_HASH_STRATEGY", "middle"),
        ] {
            let mut values = vec![("VLLM_ROUTER_L1_CACHE", "1")];
            values.retain(|(k, _)| *k != name);
            values.push((name, value));
            assert!(parse(&values).is_err(), "{name}={value}");
        }
        assert!(parse(&[
            ("VLLM_ROUTER_L1_CACHE", "1"),
            ("VLLM_ROUTER_L1_MAX_BYTES", "10"),
            ("VLLM_ROUTER_L1_MAX_INSERT_BYTES", "11"),
        ])
        .is_err());
    }

    fn long_chat_turn(i: usize) -> String {
        format!(
            "<s>user {i}: {}</s>\n",
            "lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(8)
        )
    }

    fn timings(mut f: impl FnMut()) -> (std::time::Duration, std::time::Duration) {
        let mut samples: Vec<_> = (0..30)
            .map(|_| {
                let start = std::time::Instant::now();
                f();
                start.elapsed()
            })
            .collect();
        samples.sort();
        (samples[0], samples[samples.len() / 2])
    }

    fn percentiles(mut samples: Vec<std::time::Duration>) -> String {
        samples.sort();
        let at = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
        format!(
            "p50 {:?} p99 {:?} max {:?} (n={})",
            at(0.5),
            at(0.99),
            at(1.0),
            samples.len()
        )
    }

    /// Synthetic ~440 KB chat with 1,862 boundaries built over 118 growing
    /// turns, the shape of the 128k-token review probe.
    ///
    /// `cargo test --release --lib l1_long_chat_report -- --ignored --nocapture`
    #[test]
    #[ignore = "CPU-only performance report; run in release mode"]
    fn l1_long_chat_report() {
        use super::super::encode_cache::{FrontendCache, PromptEncoder};
        use crate::tokenizer::TokenizerCacheConfig;
        use vllm_text::Prompt;

        let tokenizer = Arc::new(fixture_tokenizer());
        let turns: Vec<String> = (0..931).map(long_chat_turn).collect();
        let prompt = |n: usize| format!("{}<s>assistant\n", turns[..n].concat());
        let full = prompt(931);
        let config = L1CacheConfig {
            max_bytes: 1 << 30,
            max_insert_bytes: 1 << 28,
            ..Default::default()
        };
        println!(
            "prompt bytes {} boundaries {} tokens {}",
            full.len(),
            encoder(config).0.matcher.boundaries(&full).len(),
            tokenizer.encode(&full, false).unwrap().len()
        );

        let (single_min, _) = timings(|| {
            std::hint::black_box(tokenizer.encode(&full, false).unwrap());
        });
        let mut cold = Vec::new();
        for _ in 0..5 {
            let (encoder, _) = encoder(config);
            let start = std::time::Instant::now();
            let ids = encoder.encode(tokenizer.as_ref(), 0, &full, None).unwrap();
            cold.push(start.elapsed());
            assert_eq!(ids.unwrap().0, tokenizer.encode(&full, false).unwrap());
        }
        cold.sort();
        println!("cold: L1 {:?} vs single-pass {single_min:?}", cold[0]);

        let (encoder, cache) = encoder(config);
        for step in 1..=118 {
            check(&encoder, tokenizer.as_ref(), &prompt(931 * step / 118));
        }
        let warm = cache.stats();
        let (min, median) = timings(|| {
            encoder.encode(tokenizer.as_ref(), 0, &full, None).unwrap();
        });
        println!(
            "full hit: min {min:?} median {median:?} (nodes {} bytes {})",
            warm.nodes, warm.bytes
        );
        let (min, median) = timings(|| {
            std::hint::black_box(encoder.matcher.boundaries(&full));
        });
        println!("boundaries(full) alone: min {min:?} median {median:?}");
        let mut extra = 0;
        let (min, median) = timings(|| {
            extra += 1;
            let text = format!("{}<s>user extra {extra}</s>\n<s>assistant\n", full);
            encoder.encode(tokenizer.as_ref(), 0, &text, None).unwrap();
        });
        println!("growth by one turn: min {min:?} median {median:?}");

        let l0 = Arc::new(
            FrontendCache::new(TokenizerCacheConfig {
                max_entries: 1024,
                max_bytes: 1 << 30,
                max_entry_bytes: 1 << 26,
            })
            .unwrap(),
        );
        let both = PromptEncoder::new("fixture", tokenizer.clone(), Some(l0))
            .with_l1_cache(Some(encoder.clone()));
        let run = |text: &str| {
            both.encode_detailed(Prompt::Text(text.to_string()), false, None)
                .unwrap()
        };
        run(&full);
        let (min, median) = timings(|| {
            run(&full);
        });
        println!("L0+L1, L0 hit: min {min:?} median {median:?}");
        let mut extra = 0;
        let (min, median) = timings(|| {
            extra += 1;
            run(&format!("{full}<s>user l0 miss {extra}</s>\n"));
        });
        println!("L0+L1, L0 miss then L1 hit: min {min:?} median {median:?}");

        for threads in [1, 8, 16] {
            let start = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    let (encoder, tokenizer, full) = (&encoder, tokenizer.clone(), &full);
                    scope.spawn(move || {
                        for _ in 0..20 {
                            encoder.encode(tokenizer.as_ref(), 0, full, None).unwrap();
                        }
                    });
                }
            });
            let per = start.elapsed() / (threads * 20) as u32;
            println!("{threads} threads full hit: {per:?} wall per request");
        }

        for max_nodes in [2048, 65_536] {
            let (encoder, cache) = super::tests::encoder(L1CacheConfig {
                max_nodes,
                ..config
            });
            let (mut samples, mut alone) = (Vec::new(), Vec::new());
            for i in 0..max_nodes * 3 {
                let text = format!("<s>sys {}</s><s>user {i}</s><s>a", i % 3);
                let start = std::time::Instant::now();
                encoder.encode(tokenizer.as_ref(), 0, &text, None).unwrap();
                let elapsed = start.elapsed();
                if cache.stats().evictions > 0 {
                    samples.push(elapsed);
                    let start = std::time::Instant::now();
                    std::hint::black_box(tokenizer.encode(&text, false).unwrap());
                    alone.push(start.elapsed());
                }
            }
            println!(
                "max_nodes {max_nodes} encode once evicting: {}; tokenizer alone: {}",
                percentiles(samples),
                percentiles(alone)
            );
        }
    }
}
