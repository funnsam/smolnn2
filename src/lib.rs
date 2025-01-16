#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// layers
pub mod activation;
pub mod linear;

// optimizers
pub mod adam;
pub mod gd;

// misc
pub mod loss;

use smolmatrix::*;

#[cfg(feature = "macro")]
pub use smolnn2_macro::model;

pub trait Layer<I: Dimension, O: Dimension>: Collectable {
    /// Does forward propagation
    fn forward(&self, input: &Tensor<I>) -> Tensor<O>;

    fn derivative(
        &self,
        dc: &Tensor<O>,
        input: &Tensor<I>,
    ) -> Tensor<I>;

    fn back_propagate(
        &self,
        collector: &mut Self::Collector,
        last_derivative: Tensor<O>,
        input: &Tensor<I>,
    );
}

pub trait Collectable {
    type Collector;
}

impl<T: Collectable> Collectable for Box<T> {
    type Collector = T::Collector;
}

pub trait Optimizer<D: Dimension> {
    fn update(&mut self, w: &mut Tensor<D>, g: Tensor<D>);
}

impl<D: Dimension, T: Optimizer<D>> Optimizer<D> for Box<T> {
    fn update(&mut self, w: &mut Tensor<D>, g: Tensor<D>) {
        (**self).update(w, g);
    }
}
