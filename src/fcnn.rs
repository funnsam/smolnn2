use crate::*;

#[derive(Debug, Clone)]
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

    // fn collect(&mut self, collector: Self::Collector, optimizers: Self::Optimizers) {
    //     self.weight -= &(collector.weight * (learning_rate * (1.0 / data as f32)));
    //     self.bias -= &(collector.bias * (learning_rate * (1.0 / data as f32)));
    // }
}

impl<const I: usize, const O: usize> FcnnCollector<I, O> {
    pub fn new() -> Self {
        Self {
            weight: Matrix::new_zeroed(),
            bias: Matrix::new_zeroed(),
        }
    }
}

#[test]
fn fcnn_derivative() {
    let fcnn = Fcnn {
        weight: matrix!(1 x 1 [3.0]),
        bias: vector!(1 [0.0]),
    };
    let d = fcnn.derivative(&matrix!(1 x 1 [3.0]))[0];
    assert!((d - 9.0).abs() < f32::EPSILON, "{d}");
}
