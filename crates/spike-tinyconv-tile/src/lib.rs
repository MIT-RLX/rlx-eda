//! `spike-tinyconv-tile` — custom analog MAC tile, the first ML target
//! of the TinyConv-MNIST silicon flow.
//!
//! See `eda-bench-tinyconv/PLAN.md` build-order step 3 + co-design
//! optimization section.

pub mod behavioral;
pub mod layout;
pub mod model_card;
pub mod noise;
pub mod schem;
pub mod tile;
pub mod topology;

pub use behavioral::{
    delay_per_cycle_normalized, silicon_time_ns_per_inference, LossWeights,
};
pub use model_card::ModelCard;
pub use noise::{NoiseModel, NoiseStats};
pub use tile::{Mac8x8Tile, TileParams};
pub use topology::MacTopology;
