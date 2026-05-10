//! `Mac8x8Tile` — the parametric MAC tile.
//!
//! `TileParams` carries the variables the inner Adam loop optimizes
//! over (continuous):
//!   - `w_l_n`, `w_l_p`  — NMOS / PMOS sizing (W/L)
//!   - `vdd`             — supply
//!   - `bias_v`          — analog bias point (relevant for analog
//!                         topologies; unused by the digital default)
//!   - `weight_bits`     — discrete; set by the outer DADO loop
//!
//! `topology` is picked at construction. Default is
//! [`MacTopology::Digital`], which is the safest v1 baseline; analog
//! variants stay stubbed until someone signs up.
//!
//! `Mac8x8Tile` is one *instance-site* (designator + params +
//! topology), so `Block + Eq + Hash` works the obvious way. The PDK
//! is **not** a field — it enters via the generic `Layout<P>` /
//! `Tile<P>` impls in `layout.rs`, so the same `Mac8x8Tile` value
//! lays out under any `MosfetPdk` (sky130 default; gf180mcu and
//! beyond come for free).

use eda_hir::Block;
use serde::{Deserialize, Serialize};

use crate::topology::MacTopology;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TileParams {
    pub w_l_n: f64,
    pub w_l_p: f64,
    pub vdd: f64,
    pub bias_v: f64,
    pub weight_bits: u8,
}

impl Default for TileParams {
    /// Conservative nominal point — picked so the Adam loop has a
    /// reasonable starting basin for the digital topology under
    /// sky130. Analog topologies should override.
    fn default() -> Self {
        Self {
            w_l_n: 0.42,
            w_l_p: 0.84,
            vdd: 1.8,
            bias_v: 0.0,
            weight_bits: 8,
        }
    }
}

// f64 isn't Hash/Eq; hash bit patterns. Optimizer never passes NaN;
// if it does we want loud failure, not silent dedup.
impl Eq for TileParams {}
impl std::hash::Hash for TileParams {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.w_l_n.to_bits().hash(h);
        self.w_l_p.to_bits().hash(h);
        self.vdd.to_bits().hash(h);
        self.bias_v.to_bits().hash(h);
        self.weight_bits.hash(h);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Mac8x8Tile {
    pub instance_id: String,
    pub params: TileParams,
    pub topology: MacTopology,
}

impl Mac8x8Tile {
    /// Construct with the v1 default topology ([`MacTopology::Digital`])
    /// and given params.
    pub fn new(instance_id: impl Into<String>, params: TileParams) -> Self {
        Self::with_topology(instance_id, params, MacTopology::default())
    }

    /// Construct with an explicit topology — use when the outer DADO
    /// loop or the experimenter wants to pick a non-default flavor.
    pub fn with_topology(
        instance_id: impl Into<String>,
        params: TileParams,
        topology: MacTopology,
    ) -> Self {
        Self {
            instance_id: instance_id.into(),
            params,
            topology,
        }
    }
}

impl Block for Mac8x8Tile {
    fn name(&self) -> String {
        format!(
            "Mac8x8_{}_{}_w{}p{}_v{:.2}_b{}",
            self.topology.tag(),
            self.instance_id,
            (self.params.w_l_n * 1e3) as i64,
            (self.params.w_l_p * 1e3) as i64,
            self.params.vdd,
            self.params.weight_bits,
        )
    }
}
