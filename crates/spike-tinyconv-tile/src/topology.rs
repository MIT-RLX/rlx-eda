//! `MacTopology` — the strategy axis the inner Adam loop is *not*
//! optimizing over. Picked at construction time, fixed for a given
//! `Mac8x8Tile` instance; switch by constructing with a different
//! variant.
//!
//! Default for v1 is [`MacTopology::Digital`] because:
//!   - matches `rlx-fpga`'s existing INT8×INT8 mac.rs primitive, so
//!     the IR lowering in `spike-tinyconv-array` is direct;
//!   - synthesizable, so ORFS handles it as ground truth without
//!     mixed-signal trickery;
//!   - all four trait obligations have unambiguous bodies (well-typed
//!     even if unimplemented).
//!
//! Analog topologies are accepted as variants but their bodies stay
//! `unimplemented!()` until someone signs up for the analog design
//! work. They're declared now so the API surface doesn't churn when
//! they land.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum MacTopology {
    /// Synthesized integer MAC. Default in v1.
    Digital,
    /// Charge-redistribution analog MAC. Stub.
    ChargeRedistribution,
    /// Current-mode analog MAC. Stub.
    CurrentMode,
}

impl Default for MacTopology {
    fn default() -> Self {
        MacTopology::Digital
    }
}

impl MacTopology {
    /// Short identifier used in cell names + diagnostics.
    pub fn tag(self) -> &'static str {
        match self {
            MacTopology::Digital => "dig",
            MacTopology::ChargeRedistribution => "chr",
            MacTopology::CurrentMode => "cmd",
        }
    }
}
