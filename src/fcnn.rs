use crate::*;

#[macro_export]
macro_rules! fcnn_make_optimizers {
    ($opt:expr) => { ($opt, $opt) };
}

pub use fcnn_make_optimizers as make_optimizers;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fcnn<const I: usize, const O: usize> {
    pub weight: Matrix<I, O>,
    pub bias: Vector<O>,
}

#[derive(Debug, Clone)]
pub struct FcnnCollector<const I: usize, const O: usize> {
    pub weight: Matrix<I, O>,
    pub bias: Vector<O>,
}

impl<const I: usize, const O: usize> Fcnn<I, O> {
    pub fn new<F: Fn() -> f32>(f: F) -> Self {
        Self {
            weight: Matrix::new_zeroed().map_each(|i| *i = f()),
            bias: Matrix::new_zeroed().map_each(|i| *i = f()),
        }
    }

    /// Get uniform distributed random `f32`s in the range 0..1 and return a fully connected neural
    /// network layer with Xavier uniform initialization
    pub fn new_xavier_uniform<F: Fn() -> f32>(f: F) -> Self {
        let r = (6.0_f32 / (I + O) as f32).sqrt();

        Self {
            weight: Matrix::new_zeroed().map_each(|i| *i = f() * 2.0 * r - r),
            bias: Matrix::new_zeroed().map_each(|i| *i = f() * 2.0 * r - r),
        }
    }

    /// Get uniform distributed random `f32`s in the range 0..1 and return a fully connected neural
    /// network layer with He uniform initialization
    pub fn new_he_uniform<F: Fn() -> f32>(f: F) -> Self {
        let r = (3.0_f32 / I as f32).sqrt();

        Self {
            weight: Matrix::new_zeroed().map_each(|i| *i = f() * 2.0 * r - r),
            bias: Matrix::new_zeroed().map_each(|i| *i = f() * 2.0 * r - r),
        }
    }

    pub fn update<
        W: Optimizer<I, O>,
        B: Optimizer<1, O>,
    >(&mut self, c: &FcnnCollector<I, O>, opts: &mut (W, B)) {
        opts.0.update(&mut self.weight, c.weight.clone());
        opts.1.update(&mut self.bias  , c.bias  .clone());
    }
}

impl<const I: usize, const O: usize> Collectable for Fcnn<I, O> {
    type Collector = FcnnCollector<I, O>;
}

impl<const I: usize, const O: usize> Layer<1, I, 1, O> for Fcnn<I, O> {
    fn forward(&self, input: &Matrix<1, I>) -> Matrix<1, O> {
        &self.weight * input + &self.bias
    }

    fn derivative(
        &self,
        dc: &Matrix<1, O>,
        _input: &Matrix<1, I>,
    ) -> Matrix<1, I> {
        &self.weight.transpose() * &dc
    }

    fn back_propagate(
        &self,
        collector: &mut Self::Collector,
        last_derivative: Matrix<1, O>,
        input: &Matrix<1, I>,
    ) {
        collector.bias += &last_derivative;
        collector.weight += &(&last_derivative * &input.transpose());
    }
}

impl<const I: usize, const O: usize> FcnnCollector<I, O> {
    pub fn new() -> Self {
        Self {
            weight: Matrix::new_zeroed(),
            bias: Matrix::new_zeroed(),
        }
    }

    pub fn reset(&mut self) {
        self.weight.inner.fill([0.0; I]);
        self.bias.inner.fill([0.0]);
    }
}

impl<const I: usize, const O: usize> core::ops::DivAssign<f32> for FcnnCollector<I, O> {
    fn div_assign(&mut self, rhs: f32) {
        self.weight /= rhs;
        self.bias /= rhs;
    }
}
