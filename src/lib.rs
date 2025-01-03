// layers
pub mod activation;
pub mod fcnn;

// optimizers
pub mod adam;
pub mod gd;

// misc
pub mod loss;

use smolmatrix::*;

#[cfg(feature = "macro")]
pub use smolnn2_macro::model;

pub trait Layer<const IW: usize, const IH: usize, const OW: usize, const OH: usize>: Collectable {
    fn forward(&self, input: &Matrix<IW, IH>) -> Matrix<OW, OH>;

    fn derivative(
        &self,
        dc: &Matrix<OW, OH>,
        input: &Matrix<IW, IH>,
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

impl<T: Collectable> Collectable for Box<T> {
    type Collector = T::Collector;
}

pub trait Optimizer<const W: usize, const H: usize> {
    fn update(&mut self, w: &mut Matrix<W, H>, g: Matrix<W, H>);
}

impl<T: Optimizer<W, H>, const W: usize, const H: usize> Optimizer<W, H> for Box<T> {
    fn update(&mut self, w: &mut Matrix<W, H>, g: Matrix<W, H>) {
        (**self).update(w, g);
    }
}
