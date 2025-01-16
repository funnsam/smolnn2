use crate::*;

macro_rules! activation {
    ($name:tt $fmla:expr, $der:expr) => {
        impl Collectable for $name {
            type Collector = ();
        }

        impl<D: Dimension> Layer<D, D> for $name {
            fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
                input.clone().map_each(|i| $fmla(self, i))
            }

            fn derivative(
                &self,
                dc: &Tensor<D>,
                input: &Tensor<D>,
            ) -> Tensor<D> {
                dc.clone().map_zip_ref(input, |i, input| i * $der(self, input))
            }

            fn back_propagate(
                &self,
                _collector: &mut Self::Collector,
                _derivative: Tensor<D>,
                _input: &Tensor<D>,
            ) {}
        }
    };
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Clipped ReLU activation function (`min(max(x, 0), self.0)`)
pub struct ClippedRelu(f32);
activation!(ClippedRelu
    |s: &Self, i: f32| i.min(s.0).max(0.0),
    |s: &Self, i: f32| if 0.0 < i && i < s.0 { 1.0 } else { 0.0 }
);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Leaky ReLU activation function for `0 <= self.0 <= 1` (`max(x, self.0 * x)`)
pub struct LeakyRelu(pub f32);
activation!(LeakyRelu
    |s: &Self, i: f32| i.max(s.0 * i),
    |s: &Self, i: f32| if 0.0 <= i { 1.0 } else { s.0 }
);

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// `tanh` activation function
pub struct Tanh;
activation!(Tanh
    |_, i: f32| i.tanh(),
    |_, i: f32| i.cosh().powi(2).recip()
);

/// Normalized softmax activation function
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Softmax;

impl Collectable for Softmax {
    type Collector = ();
    }

impl<D: Dimension> Layer<D, D> for Softmax {
    fn forward(&self, input: &Tensor<D>) -> Tensor<D> {
        let mut input = input.clone();
        let max = input.inner.as_ref().iter().fold(f32::NEG_INFINITY, |c, i| c.max(*i));
        input.map_each_in_place(|i| (i - max).exp());
        let sum = input.inner.as_ref().iter().sum::<f32>();
        input / sum
    }

    fn derivative(
        &self,
        dc: &Tensor<D>,
        _input: &Tensor<D>,
    ) -> Tensor<D> {
        dc.clone()
    }

    fn back_propagate(
        &self,
        _collector: &mut Self::Collector,
        _derivative: Tensor<D>,
        _input: &Tensor<D>,
    ) {}
}
