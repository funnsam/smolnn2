// layers
pub mod activation;
pub mod fcnn;

// optimizers
pub mod adam;
pub mod gd;

use smolmatrix::*;

#[cfg(feature = "macro")]
pub use smolnn2_macro::model;

pub trait Layer<const IW: usize, const IH: usize, const OW: usize, const OH: usize>: Collectable {
    fn forward(&self, input: &Matrix<IW, IH>) -> Matrix<OW, OH>;

    fn derivative(
        &self,
        dc: &Matrix<OW, OH>,
    ) -> Matrix<IW, IH>;

    fn back_propagate(
        &self,
        collector: &mut Self::Collector,
        last_derivative: Matrix<OW, OH>,
        input: &Matrix<IW, IH>,
    );
}

pub trait Collectable {
    type Collector;
}

pub trait Optimizer<const W: usize, const H: usize> {
    fn update(&mut self, w: &mut Matrix<W, H>, g: Matrix<W, H>);
}

pub fn squared_error<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    (result - expected).map_each(|i| *i *= *i)
}

pub fn squared_error_derivative<const W: usize, const H: usize>(
    result: Matrix<W, H>,
    expected: &Matrix<W, H>,
) -> Matrix<W, H> {
    (result - expected) * 2.0
}
