//! Fujifilm compressed RAF image data: the "lossless compressed" raw recording mode of X-Trans and Bayer bodies.
//!
//! **Provenance: not clean-room.** Every other decoder in this crate was written from prose descriptions and
//! black-box analysis of samples (see `nefc.rs`). No permissively licensed description of Fujifilm's compression
//! exists. The format knowledge here comes from the AI model that wrote this module, and that model's training
//! data includes published decoder source code (LibRaw and rawspeed, both LGPL). The code is an independent Rust
//! implementation (its own bit reader, a table of decoding passes, its own data layout), not a translation of any
//! source file. It still does **not** meet the project's clean-room rule (CLAUDE.md, ROADMAP 2026-10-05). It was
//! added at the repository owner's request (2026-10-08) so that their camera's files open, and must be reviewed
//! against that rule before it is merged upstream.
//!
//! Checked on a real X-T20 RAF (lossless compressed, 14-bit X-Trans III, 6048×4038, 8 blocks of 768 pixels): every
//! block's code stream ends 3–9 bytes before its declared size, followed only by zero padding (a decoding error
//! anywhere would desynchronise the stream long before that); the active area holds no out-of-range samples, block
//! boundaries show no seams, and the developed image matches the camera's embedded JPEG (mean CIE76 ΔE 2.5). The
//! round trip through the test encoder below covers Bayer data, 12/14/16-bit samples and partial blocks; Bayer
//! (GFX) files have not been checked on real data.
//!
//! Format:
//! - The raw strip (`0xf007`/`0xf008`) starts with a 16-byte big-endian header: signature `IS`, lossless flag,
//!   sensor kind (16 = X-Trans, 0 = Bayer), bits per sample, height, width rounded up to whole blocks, width,
//!   block width, blocks per row and the number of six-row groups. One `u32` byte size per block follows, padded to
//!   a multiple of 16 bytes, then the blocks back to back.
//! - Each block is an independent vertical strip of the sensor (the last one partly past the image width), coded six
//!   rows at a time. A group's samples are sorted into colour lines: red 3, green 6, blue 3, each 2/3 of the block
//!   wide (X-Trans) or 1/2 (Bayer). On X-Trans some line positions hold no sensor sample; they are interpolated
//!   instead of coded ([`Even`]). Two lines per colour are kept from the previous group as context.
//! - The lines are coded in six passes of two lines each ([`XTRANS`], [`BAYER`]): even positions run ahead, odd
//!   positions follow nine samples behind and are predicted from both neighbours. Each sample is predicted from
//!   the lines above; the residual is coded with an adaptive Golomb–Rice code whose parameter comes from one of 41
//!   contexts (quantised local gradients), with an escape to a raw `bits`-bit value.
//! - Lossy compressed RAF (a quantisation table per line) is not decoded: [`RawError::Unsupported`], so the
//!   embedded preview is shown.

use crate::{Cfa, MAX_SAMPLES, RawError, Result};
use rayon::prelude::*;

const HEADER_LEN: usize = 16;

fn corrupt(why: &str) -> RawError {
    RawError::Corrupt(format!("compressed RAF: {why}"))
}

fn be16(b: &[u8], at: usize) -> usize {
    b.get(at..at + 2).map_or(0, |s| usize::from(u16::from_be_bytes([s[0], s[1]])))
}

/// The strip header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub lossless: bool,
    pub xtrans: bool,
    pub bits: u32,
    pub width: usize,
    pub height: usize,
    block: usize,
    blocks: usize,
    groups: usize,
}

impl Header {
    /// Whether `strip` starts like a compressed strip (the signature only).
    pub(crate) fn signature(strip: &[u8]) -> bool {
        strip.starts_with(b"IS")
    }

    pub(crate) fn parse(strip: &[u8]) -> Result<Header> {
        let h = strip.get(..HEADER_LEN).ok_or_else(|| corrupt("header truncated"))?;
        if !Header::signature(h) {
            return Err(RawError::Unsupported("Fujifilm compressed RAF with an unknown header".into()));
        }
        let (lossless, kind, bits) = (h[2], h[3], u32::from(h[4]));
        if lossless > 1 || !matches!(kind, 0 | 16) || !matches!(bits, 12 | 14 | 16) {
            return Err(RawError::Unsupported(format!("Fujifilm compressed RAF variant (flag {lossless}, kind {kind}, {bits}-bit)")));
        }
        let header = Header {
            lossless: lossless == 1,
            xtrans: kind == 16,
            bits,
            width: be16(h, 9),
            height: be16(h, 5),
            block: be16(h, 11),
            blocks: usize::from(h[13]),
            groups: be16(h, 14),
        };
        let rounded = be16(h, 7);
        let unit = if header.xtrans { 6 } else { 2 };
        let lw = header.line_width();
        if header.width == 0
            || header.height == 0
            || header.block == 0
            || !header.block.is_multiple_of(unit)
            || lw < 16
            || !lw.is_multiple_of(2)
            || header.blocks == 0
            || rounded != header.blocks * header.block
            || rounded < header.width
            || rounded - header.width >= header.block
            || header.groups != header.height.div_ceil(6)
        {
            return Err(corrupt(&format!("inconsistent header {header:?}, rounded width {rounded}")));
        }
        if header.width.saturating_mul(header.height) > MAX_SAMPLES {
            return Err(RawError::Limit("image too large"));
        }
        Ok(header)
    }

    /// Samples per colour line.
    fn line_width(&self) -> usize {
        if self.xtrans { self.block / 3 * 2 } else { self.block / 2 }
    }

    /// The coded bytes of each block (validates the block table against the strip).
    pub(crate) fn blocks<'a>(&self, strip: &'a [u8]) -> Result<Vec<&'a [u8]>> {
        let table = strip.get(HEADER_LEN..HEADER_LEN + 4 * self.blocks).ok_or_else(|| corrupt("block table truncated"))?;
        let mut at = HEADER_LEN + (4 * self.blocks).next_multiple_of(16);
        table
            .chunks_exact(4)
            .map(|c| {
                let size = u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as usize;
                let end = at.checked_add(size).ok_or_else(|| corrupt("block size overflows"))?;
                let block = strip.get(at..end).ok_or_else(|| corrupt("block past the end of the strip"))?;
                at = end;
                Ok(block)
            })
            .collect()
    }
}

/// Decode the strip into `header.width × header.height` samples laid out like the sensor, with `cfa` (anchored at
/// raw pixel (0, 0)) deciding which colour line each pixel comes from.
pub(crate) fn decode(strip: &[u8], header: &Header, cfa: &Cfa) -> Result<Vec<u16>> {
    if !header.lossless {
        return Err(RawError::Unsupported("Fujifilm lossy compressed RAF".into()));
    }
    let size = if header.xtrans { 6 } else { 2 };
    if cfa.width != size || cfa.height != size || !cfa.valid() {
        return Err(corrupt("colour filter layout does not match the sensor kind"));
    }
    let blocks = header.blocks(strip)?;
    let params = Params::new(header.bits);
    let decoded: Vec<Result<Vec<u16>>> =
        blocks.par_iter().enumerate().map(|(index, data)| Block::new(header, &params, data).decode(header, index, cfa)).collect();
    let (w, h) = (header.width, header.height);
    let mut out = vec![0u16; w * h];
    for (index, block) in decoded.into_iter().enumerate() {
        let block = block?;
        let x0 = index * header.block;
        let bw = header.block.min(w.saturating_sub(x0));
        if bw == 0 {
            continue;
        }
        for (row, src) in out.chunks_exact_mut(w).zip(block.chunks_exact(bw)) {
            if let Some(dst) = row.get_mut(x0..x0 + bw) {
                dst.copy_from_slice(src);
            }
        }
    }
    Ok(out)
}

/// Constants of one bit depth.
struct Params {
    bits: u32,
    /// Largest sample value.
    max: i32,
    /// `max + 1`: residuals wrap around modulo this.
    total: i32,
    /// A run of this many zero bits (or more) escapes to a raw `bits`-bit code.
    escape: u32,
    /// Initial context mean.
    initial: u32,
    /// Gradient quantiser: `quant[d + max]` in −4..=4.
    quant: Vec<i8>,
}

/// Gradient quantisation thresholds.
const STEPS: [i32; 3] = [0x12, 0x43, 0x114];

impl Params {
    fn new(bits: u32) -> Params {
        let total = 1i32 << bits;
        let max = total - 1;
        let quant = (-max..=max)
            .map(|d| {
                let a = d.abs();
                let level = if a == 0 { 0 } else { 1 + STEPS.iter().filter(|&&s| a >= s).count() as i8 };
                level * d.signum() as i8
            })
            .collect();
        Params { bits, max, total, escape: 3 * bits - 1, initial: ((total as u32 + 32) >> 6).max(2), quant }
    }

    fn quant(&self, d: i32) -> i32 {
        usize::try_from(d + self.max).ok().and_then(|i| self.quant.get(i)).map_or(0, |&q| i32::from(q))
    }

    /// The sample `predicted` ± the residual, wrapped into the sample range.
    fn sample(&self, predicted: i32, grad: i32, diff: i32) -> u16 {
        let v = if grad < 0 { predicted - diff } else { predicted + diff };
        let v = if v < 0 {
            v + self.total
        } else if v > self.max {
            v - self.total
        } else {
            v
        };
        v.clamp(0, self.max) as u16
    }

    /// The gradient context of a sample (0..41) and its sign.
    fn context(&self, a: i32, b: i32) -> i32 {
        9 * self.quant(a) + self.quant(b)
    }
}

/// An adaptive Golomb–Rice context: running sum of residual magnitudes and their count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Context {
    sum: u32,
    count: u32,
}

/// The count at which a context's sums are halved.
const HALVE_AT: u32 = 64;

impl Context {
    /// Number of low bits sent verbatim: the smallest `k` (at most 15) with `count << k >= sum`.
    fn shift(self) -> u32 {
        let mut k = 0;
        if self.count < self.sum {
            while k <= 14 {
                k += 1;
                if (self.count << k) >= self.sum {
                    break;
                }
            }
        }
        k
    }

    fn update(&mut self, magnitude: u32) {
        self.sum += magnitude;
        if self.count == HALVE_AT {
            self.sum >>= 1;
            self.count >>= 1;
        }
        self.count += 1;
    }
}

/// Map a Golomb code to a signed residual (even codes ≥ 0, odd codes < 0).
fn unzigzag(code: u32) -> i32 {
    let half = (code >> 1) as i32;
    if code & 1 == 1 { -1 - half } else { half }
}

/// MSB-first bit reader over one block; reads past the end give zero bits (and an error once a zero run is
/// impossibly long).
struct Bits<'a> {
    data: &'a [u8],
    next: usize,
    cache: u64,
    len: u32,
}

/// Longest zero run accepted before a block is reported corrupt (valid escapes use fewer than 48).
const MAX_ZEROS: u32 = 256;

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Bits<'a> {
        Bits { data, next: 0, cache: 0, len: 0 }
    }

    fn refill(&mut self) {
        while self.len <= 56 {
            let byte = self.data.get(self.next).copied().unwrap_or(0);
            self.cache |= u64::from(byte) << (56 - self.len);
            self.len += 8;
            self.next += 1;
        }
    }

    /// `n ≤ 32` bits.
    fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.len < n {
            self.refill();
        }
        let v = (self.cache >> (64 - n)) as u32;
        self.cache = self.cache.checked_shl(n).unwrap_or(0);
        self.len -= n;
        v
    }

    /// Count zero bits up to the next one bit, which is consumed.
    fn zeros(&mut self) -> Result<u32> {
        let mut count = 0;
        loop {
            self.refill();
            let lz = self.cache.leading_zeros();
            if lz < self.len {
                self.cache = self.cache.checked_shl(lz + 1).unwrap_or(0);
                self.len -= lz + 1;
                return Ok(count + lz);
            }
            count += self.len;
            self.cache = 0;
            self.len = 0;
            if count > MAX_ZEROS {
                return Err(corrupt("runaway zero run (data truncated or damaged)"));
            }
        }
    }

    /// Bytes consumed so far (rounded up).
    fn used(&self) -> usize {
        (self.next * 8 - self.len as usize).div_ceil(8)
    }
}

/// Read one residual in context `ctx`.
fn residual(bits: &mut Bits, p: &Params, ctx: &mut Context) -> Result<i32> {
    let zeros = bits.zeros()?;
    let code = if zeros < p.escape {
        let k = ctx.shift();
        (zeros << k) | bits.read(k)
    } else {
        bits.read(p.bits) + 1
    };
    if code >= p.total as u32 {
        return Err(corrupt("residual out of range"));
    }
    let diff = unzigzag(code);
    ctx.update(diff.unsigned_abs());
    Ok(diff)
}

// Colour lines of a block: per colour two lines carried over from the previous group, then the current group's.
const R2: usize = 2;
const G2: usize = 7;
const B2: usize = 15;
const LINES: usize = 18;
const RED: (usize, usize) = (R2, R2 + 2);
const GREEN: (usize, usize) = (G2, G2 + 5);
const BLUE: (usize, usize) = (B2, B2 + 2);

/// How a pass treats a line's even positions (odd positions are always coded).
#[derive(Clone, Copy, Debug)]
enum Even {
    Code,
    /// No sensor sample at any even position: interpolated from the line above.
    Interpolate,
    /// Interpolated where `position % 4` equals the value, coded elsewhere.
    InterpolateAt(usize),
}

impl Even {
    fn interpolates(self, pos: usize) -> bool {
        match self {
            Even::Code => false,
            Even::Interpolate => true,
            Even::InterpolateAt(r) => pos % 4 == r,
        }
    }
}

/// One coding pass: two lines (interleaved sample by sample), the context set they share, and the colour ranges
/// whose padding samples are refreshed afterwards.
struct Pass {
    lines: [(usize, Even); 2],
    contexts: usize,
    extend: [(usize, usize); 2],
}

/// X-Trans: which line positions hold a sensor sample follows from [`position`] and the 6×6 layout; the rest are
/// interpolated.
const XTRANS: [Pass; 6] = [
    Pass { lines: [(R2, Even::Interpolate), (G2, Even::Code)], contexts: 0, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 1, Even::Code), (B2, Even::Interpolate)], contexts: 1, extend: [GREEN, BLUE] },
    Pass { lines: [(R2 + 1, Even::InterpolateAt(0)), (G2 + 2, Even::Interpolate)], contexts: 2, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 3, Even::Code), (B2 + 1, Even::InterpolateAt(2))], contexts: 0, extend: [GREEN, BLUE] },
    Pass { lines: [(R2 + 2, Even::InterpolateAt(2)), (G2 + 4, Even::Code)], contexts: 1, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 5, Even::Interpolate), (B2 + 2, Even::InterpolateAt(0))], contexts: 2, extend: [GREEN, BLUE] },
];

/// Bayer: every line position is a sensor sample.
const BAYER: [Pass; 6] = [
    Pass { lines: [(R2, Even::Code), (G2, Even::Code)], contexts: 0, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 1, Even::Code), (B2, Even::Code)], contexts: 1, extend: [GREEN, BLUE] },
    Pass { lines: [(R2 + 1, Even::Code), (G2 + 2, Even::Code)], contexts: 2, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 3, Even::Code), (B2 + 1, Even::Code)], contexts: 0, extend: [GREEN, BLUE] },
    Pass { lines: [(R2 + 2, Even::Code), (G2 + 4, Even::Code)], contexts: 1, extend: [RED, GREEN] },
    Pass { lines: [(G2 + 5, Even::Code), (B2 + 2, Even::Code)], contexts: 2, extend: [GREEN, BLUE] },
];

/// Odd positions start once the even ones are this far ahead.
const ODD_LAG: usize = 8;

/// The colour line and position holding pixel `x` (relative to the block) of row `row` (0..6) of a group.
fn position(xtrans: bool, colour: u8, row: usize, x: usize) -> (usize, usize) {
    let line = match colour {
        0 => R2 + row / 2,
        1 => G2 + row,
        _ => B2 + row / 2,
    };
    let pos = if xtrans { 2 * (x / 3) + usize::from(!x.is_multiple_of(3)) } else { x / 2 };
    (line, pos)
}

/// The even-position predictor (also the interpolation of positions without a sensor sample): the vertical
/// neighbour weighted twice, plus the two of the other three neighbours that best follow the local edge.
fn predict_even(rb: i32, rc: i32, rd: i32, rf: i32) -> i32 {
    let (dc, df, dd) = ((rc - rb).abs(), (rf - rb).abs(), (rd - rb).abs());
    let pair = if dc > df && dc > dd {
        rf + rd
    } else if dd > dc && dd > df {
        rf + rc
    } else {
        rd + rc
    };
    (pair + 2 * rb) >> 2
}

/// The odd-position predictor from the left and right neighbours (and the line above when it is a local extreme).
fn predict_odd(ra: i32, rb: i32, rc: i32, rd: i32, rg: i32) -> i32 {
    if (rb > rc && rb > rd) || (rb < rc && rb < rd) { (rg + ra + 2 * rb) >> 2 } else { (ra + rg) >> 1 }
}

/// Decoding state of one block.
struct Block<'a> {
    bits: Bits<'a>,
    params: &'a Params,
    /// Samples per line; lines are stored with one padding sample on either side.
    width: usize,
    lines: Vec<u16>,
    even: [[Context; 41]; 3],
    odd: [[Context; 41]; 3],
}

impl<'a> Block<'a> {
    fn new(header: &Header, params: &'a Params, data: &'a [u8]) -> Block<'a> {
        let width = header.line_width();
        let start = Context { sum: params.initial, count: 1 };
        Block { bits: Bits::new(data), params, width, lines: vec![0; LINES * (width + 2)], even: [[start; 41]; 3], odd: [[start; 41]; 3] }
    }

    fn index(&self, line: usize, pos: isize) -> Option<usize> {
        let p = usize::try_from(pos + 1).ok().filter(|&p| p < self.width + 2)?;
        Some(line * (self.width + 2) + p)
    }

    fn at(&self, line: usize, pos: isize) -> i32 {
        self.index(line, pos).and_then(|i| self.lines.get(i)).map_or(0, |&v| i32::from(v))
    }

    fn set(&mut self, line: usize, pos: isize, v: u16) {
        if let Some(s) = self.index(line, pos).and_then(|i| self.lines.get_mut(i)) {
            *s = v;
        }
    }

    fn decode(mut self, header: &Header, index: usize, cfa: &Cfa) -> Result<Vec<u16>> {
        let passes = if header.xtrans { &XTRANS } else { &BAYER };
        let x0 = index * header.block;
        let bw = header.block.min(header.width.saturating_sub(x0));
        let mut out = vec![0u16; bw * header.height];
        for group in 0..header.groups {
            for pass in passes {
                self.pass(pass)?;
            }
            for row in 0..6 {
                let y = group * 6 + row;
                let Some(dst) = out.get_mut(y * bw..(y + 1) * bw) else { break };
                for (x, o) in dst.iter_mut().enumerate() {
                    let (line, pos) = position(header.xtrans, cfa.color_at(x0 + x, y), row, x);
                    *o = self.at(line, pos as isize) as u16;
                }
            }
            self.next_group();
        }
        let used = self.bits.used();
        if used > self.bits.data.len() {
            return Err(corrupt(&format!("block {index} needs {used} bytes, has {}", self.bits.data.len())));
        }
        Ok(out)
    }

    fn pass(&mut self, pass: &Pass) -> Result<()> {
        let (mut even, mut odd) = (0usize, 1usize);
        while even < self.width || odd < self.width {
            if even < self.width {
                for &(line, mode) in &pass.lines {
                    if mode.interpolates(even) {
                        self.interpolate(line, even as isize);
                    } else {
                        self.code_even(line, even as isize, pass.contexts)?;
                    }
                }
                even += 2;
            }
            if even > ODD_LAG && odd < self.width {
                for &(line, _) in &pass.lines {
                    self.code_odd(line, odd as isize, pass.contexts)?;
                }
                odd += 2;
            }
        }
        for range in pass.extend {
            self.extend(range);
        }
        Ok(())
    }

    /// Neighbours of an even position: above (b), above-left (c), above-right (d), two lines up (f).
    fn above(&self, line: usize, pos: isize) -> (i32, i32, i32, i32) {
        (self.at(line - 1, pos), self.at(line - 1, pos - 1), self.at(line - 1, pos + 1), self.at(line - 2, pos))
    }

    fn interpolate(&mut self, line: usize, pos: isize) {
        let (rb, rc, rd, rf) = self.above(line, pos);
        self.set(line, pos, predict_even(rb, rc, rd, rf) as u16);
    }

    fn code_even(&mut self, line: usize, pos: isize, set: usize) -> Result<()> {
        let (rb, rc, rd, rf) = self.above(line, pos);
        let grad = self.params.context(rb - rf, rc - rb);
        let ctx = self.even.get_mut(set).and_then(|s| s.get_mut(grad.unsigned_abs() as usize)).ok_or_else(|| corrupt("context"))?;
        let diff = residual(&mut self.bits, self.params, ctx)?;
        let v = self.params.sample(predict_even(rb, rc, rd, rf), grad, diff);
        self.set(line, pos, v);
        Ok(())
    }

    fn code_odd(&mut self, line: usize, pos: isize, set: usize) -> Result<()> {
        let (ra, rg) = (self.at(line, pos - 1), self.at(line, pos + 1));
        let (rb, rc, rd) = (self.at(line - 1, pos), self.at(line - 1, pos - 1), self.at(line - 1, pos + 1));
        let grad = self.params.context(rb - rc, rc - ra);
        let ctx = self.odd.get_mut(set).and_then(|s| s.get_mut(grad.unsigned_abs() as usize)).ok_or_else(|| corrupt("context"))?;
        let diff = residual(&mut self.bits, self.params, ctx)?;
        let v = self.params.sample(predict_odd(ra, rb, rc, rd, rg), grad, diff);
        self.set(line, pos, v);
        Ok(())
    }

    /// Refresh the padding samples of lines `first..=last` from the first and last samples of the line above.
    fn extend(&mut self, (first, last): (usize, usize)) {
        let end = self.width as isize - 1;
        for line in first..=last {
            let (l, r) = (self.at(line - 1, 0), self.at(line - 1, end));
            self.set(line, -1, l as u16);
            self.set(line, end + 1, r as u16);
        }
    }

    /// Keep each colour's last two lines as context for the next group and clear the current lines.
    fn next_group(&mut self) {
        let stride = self.width + 2;
        for (first, last) in [RED, GREEN, BLUE] {
            for k in 0..2 {
                let (src, dst) = (last - 1 + k, first - 2 + k);
                if (src + 1) * stride <= self.lines.len() {
                    self.lines.copy_within(src * stride..(src + 1) * stride, dst * stride);
                }
            }
            if let Some(current) = self.lines.get_mut(first * stride..(last + 1) * stride) {
                current.fill(0);
            }
            self.extend((first, first));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// MSB-first bit writer.
    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        acc: u64,
        n: u32,
    }

    impl Writer {
        fn put(&mut self, v: u32, n: u32) {
            for i in (0..n).rev() {
                self.acc = (self.acc << 1) | u64::from(v.checked_shr(i).unwrap_or(0) & 1);
                self.n += 1;
                if self.n == 8 {
                    self.bytes.push(self.acc as u8);
                    (self.acc, self.n) = (0, 0);
                }
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.put(0, 8 - self.n);
            }
            self.bytes
        }
    }

    fn zigzag(d: i32) -> u32 {
        if d >= 0 { 2 * d as u32 } else { (-2 * d - 1) as u32 }
    }

    /// The encoder mirroring [`Block`]: line positions with a sensor sample take the image's values, the others
    /// the same interpolation as the decoder; each coded sample's residual is wrapped to the shortest code.
    fn encode_block(header: &Header, params: &Params, cfa: &Cfa, img: &[u16], index: usize) -> Vec<u8> {
        let mut block = Block::new(header, params, &[]);
        let mut w = Writer::default();
        let passes = if header.xtrans { &XTRANS } else { &BAYER };
        let x0 = index * header.block;
        for group in 0..header.groups {
            // the image's samples of this group, by line position (rows past the image are 0)
            let mut target = vec![None; LINES * (block.width + 2)];
            for row in 0..6 {
                let y = group * 6 + row;
                for x in 0..header.block {
                    let (line, pos) = position(header.xtrans, cfa.color_at(x0 + x, y), row, x);
                    let v = if y < header.height && x0 + x < header.width { img[y * header.width + x0 + x] } else { 0 };
                    let i = block.index(line, pos as isize).unwrap();
                    assert!(target[i].is_none(), "two pixels map to line {line} position {pos}");
                    target[i] = Some(v);
                }
            }
            let mut code = |block: &mut Block, ctx: &mut Context, line: usize, pos: isize, predicted: i32, grad: i32| {
                let i = block.index(line, pos).unwrap();
                let v = target[i].unwrap_or_else(|| panic!("coded line {line} position {pos} holds no sensor sample"));
                let raw = if grad < 0 { predicted - i32::from(v) } else { i32::from(v) - predicted };
                let half = params.total / 2;
                let d = (raw + half).rem_euclid(params.total) - half;
                let c = zigzag(d);
                let k = ctx.shift();
                if (c >> k) < params.escape {
                    w.put(0, c >> k);
                    w.put(1, 1);
                    w.put(c & ((1 << k) - 1), k);
                } else {
                    w.put(0, params.escape);
                    w.put(1, 1);
                    w.put(c - 1, params.bits);
                }
                ctx.update(d.unsigned_abs());
                block.set(line, pos, v);
            };
            for pass in passes {
                let (mut even, mut odd) = (0usize, 1usize);
                while even < block.width || odd < block.width {
                    if even < block.width {
                        for &(line, mode) in &pass.lines {
                            let pos = even as isize;
                            let (rb, rc, rd, rf) = block.above(line, pos);
                            if mode.interpolates(even) {
                                assert!(target[block.index(line, pos).unwrap()].is_none(), "line {line} position {pos} holds a sensor sample");
                                block.interpolate(line, pos);
                            } else {
                                let grad = params.context(rb - rf, rc - rb);
                                let mut ctx = block.even[pass.contexts][grad.unsigned_abs() as usize];
                                code(&mut block, &mut ctx, line, pos, predict_even(rb, rc, rd, rf), grad);
                                block.even[pass.contexts][grad.unsigned_abs() as usize] = ctx;
                            }
                        }
                        even += 2;
                    }
                    if even > ODD_LAG && odd < block.width {
                        for &(line, _) in &pass.lines {
                            let pos = odd as isize;
                            let (ra, rg) = (block.at(line, pos - 1), block.at(line, pos + 1));
                            let (rb, rc, rd) = (block.at(line - 1, pos), block.at(line - 1, pos - 1), block.at(line - 1, pos + 1));
                            let grad = params.context(rb - rc, rc - ra);
                            let mut ctx = block.odd[pass.contexts][grad.unsigned_abs() as usize];
                            code(&mut block, &mut ctx, line, pos, predict_odd(ra, rb, rc, rd, rg), grad);
                            block.odd[pass.contexts][grad.unsigned_abs() as usize] = ctx;
                        }
                        odd += 2;
                    }
                }
                for range in pass.extend {
                    block.extend(range);
                }
            }
            block.next_group();
        }
        w.finish()
    }

    /// A whole strip: header, block table (padded to 16 bytes), blocks (each followed by `pad` zero bytes).
    pub(crate) fn encode(xtrans: bool, bits: u32, width: usize, height: usize, block: usize, img: &[u16], cfa: &Cfa, pad: usize) -> Vec<u8> {
        let blocks = width.div_ceil(block);
        let mut h = vec![0x49, 0x53, 1, if xtrans { 16 } else { 0 }, bits as u8];
        for v in [height, blocks * block, width, block] {
            h.extend_from_slice(&(v as u16).to_be_bytes());
        }
        h.push(blocks as u8);
        h.extend_from_slice(&(height.div_ceil(6) as u16).to_be_bytes());
        let header = Header::parse(&h).unwrap();
        let params = Params::new(bits);
        let coded: Vec<Vec<u8>> = (0..blocks)
            .map(|i| {
                let mut b = encode_block(&header, &params, cfa, img, i);
                b.resize(b.len() + pad, 0);
                b
            })
            .collect();
        for b in &coded {
            h.extend_from_slice(&(b.len() as u32).to_be_bytes());
        }
        h.resize(HEADER_LEN + (4 * blocks).next_multiple_of(16), 0);
        coded.iter().for_each(|b| h.extend_from_slice(b));
        h
    }

    /// Smooth gradients plus noise and a few hard edges and extremes, like a photo.
    pub(crate) fn image(width: usize, height: usize, bits: u32, seed: u32) -> Vec<u16> {
        let max = (1u32 << bits) - 1;
        let mut s = seed;
        (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let base = (x * 37 + y * 23) as u32 % (max / 2) + if (x / 7 + y / 5) % 3 == 0 { max / 3 } else { 0 };
                let v = if s.is_multiple_of(97) {
                    max
                } else if s.is_multiple_of(89) {
                    0
                } else {
                    (base + s % 64).min(max)
                };
                v as u16
            })
            .collect()
    }

    #[test]
    fn round_trips_xtrans_and_bayer_with_partial_blocks() {
        for (xtrans, bits, width, height, block) in
            [(true, 14, 100, 30, 48), (true, 12, 96, 12, 48), (false, 14, 70, 18, 32), (false, 12, 64, 13, 32), (true, 16, 48, 6, 24)]
        {
            let cfa = if xtrans { Cfa::xtrans() } else { Cfa::bayer_static("RGGB") };
            let img = image(width, height, bits, width as u32 * 7 + bits);
            let strip = encode(xtrans, bits, width, height, block, &img, &cfa, 3);
            let header = Header::parse(&strip).unwrap();
            assert_eq!((header.width, header.height, header.bits, header.xtrans), (width, height, bits, xtrans));
            let out = decode(&strip, &header, &cfa).unwrap_or_else(|e| panic!("{xtrans} {bits}: {e}"));
            assert_eq!(out, img, "xtrans {xtrans}, {bits}-bit, {width}×{height}, block {block}");
        }
    }

    #[test]
    fn x_trans_lines_hold_every_sensor_sample_once() {
        // every pixel of a 6-row group maps to its own line position, and a pass codes exactly those positions
        let cfa = Cfa::xtrans();
        let mut seen = std::collections::HashSet::new();
        for row in 0..6 {
            for x in 0..48 {
                let (line, pos) = position(true, cfa.color_at(x, row), row, x);
                assert!(seen.insert((line, pos)), "({x}, {row})");
            }
        }
        let lw = 32;
        let coded: usize =
            XTRANS.iter().flat_map(|p| p.lines).map(|(_, mode)| (0..lw).filter(|&pos| pos % 2 == 1 || !mode.interpolates(pos)).count()).sum();
        assert_eq!(coded, seen.len());
        for &(line, pos) in &seen {
            let mode = XTRANS.iter().flat_map(|p| p.lines).find(|l| l.0 == line).unwrap().1;
            assert!(pos % 2 == 1 || !mode.interpolates(pos), "sample at line {line} position {pos} is interpolated");
        }
    }

    #[test]
    fn rejects_bad_headers_and_damaged_data_without_panicking() {
        let cfa = Cfa::xtrans();
        let img = image(96, 12, 14, 3);
        let strip = encode(true, 14, 96, 12, 48, &img, &cfa, 0);
        assert!(matches!(Header::parse(b"IS"), Err(RawError::Corrupt(_))));
        assert!(matches!(Header::parse(&[0u8; 16]), Err(RawError::Unsupported(_))));
        for (at, value) in [(2usize, 2u8), (3, 5), (4, 10)] {
            let mut s = strip.clone();
            s[at] = value;
            assert!(matches!(Header::parse(&s), Err(RawError::Unsupported(_))), "byte {at} = {value}");
        }
        for (at, value) in [(6usize, 0u8), (12, 0), (13, 0), (13, 9), (15, 9), (12, 7), (10, 0x61)] {
            let mut s = strip.clone();
            s[at] = value;
            assert!(Header::parse(&s).is_err(), "byte {at} = {value}");
        }
        let header = Header::parse(&strip).unwrap();
        // lossy, mismatched layout
        let mut lossy = strip.clone();
        lossy[2] = 0;
        assert!(matches!(decode(&lossy, &Header::parse(&lossy).unwrap(), &cfa), Err(RawError::Unsupported(_))));
        assert!(decode(&strip, &header, &Cfa::bayer_static("RGGB")).is_err());
        // truncated strip, garbage and all-zero blocks: errors, never panics or hangs
        assert!(decode(&strip[..strip.len() - 20], &header, &cfa).is_err());
        assert!(decode(&strip[..20], &header, &cfa).is_err());
        let data = HEADER_LEN + 16;
        let mut zeros = strip.clone();
        zeros[data..].fill(0);
        assert!(decode(&zeros, &header, &cfa).is_err());
        let mut s = 0x9e37_79b9u32;
        for _ in 0..20 {
            let mut garbage = strip.clone();
            for b in &mut garbage[data..] {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = s as u8;
            }
            let _ = decode(&garbage, &header, &cfa);
        }
    }

    #[test]
    fn context_shift_and_update() {
        assert_eq!(Context { sum: 1, count: 1 }.shift(), 0);
        assert_eq!(Context { sum: 256, count: 1 }.shift(), 8);
        assert_eq!(Context { sum: 257, count: 1 }.shift(), 9);
        assert_eq!(Context { sum: u32::MAX, count: 1 }.shift(), 15);
        let mut c = Context { sum: 100, count: HALVE_AT };
        c.update(10);
        assert_eq!(c, Context { sum: 55, count: HALVE_AT / 2 + 1 });
        assert_eq!((unzigzag(0), unzigzag(1), unzigzag(2), unzigzag(3)), (0, -1, 1, -2));
        let p = Params::new(14);
        assert_eq!((p.quant(0), p.quant(17), p.quant(18), p.quant(-67), p.quant(276), p.quant(-16383)), (0, 1, 2, -3, 4, -4));
        assert_eq!((p.escape, p.initial), (41, 256));
        assert_eq!(p.sample(10, 1, -20), 16374);
        assert_eq!(p.sample(16380, -1, -10), 6);
    }
}
