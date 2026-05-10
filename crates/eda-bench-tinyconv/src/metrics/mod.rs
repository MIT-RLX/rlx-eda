//! Two metric arms — both required to pass before any candidate is
//! reported. See PLAN.md "Validation" + "Bench harness layout".

pub mod functional;
pub mod physical;

pub use functional::Functional;
pub use physical::Physical;
