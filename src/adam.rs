use crate::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Adam<D: Dimension> {
    pub beta_1: f32,
    pub beta_2: f32,
    pub eta: f32,
    m: Tensor<D>,
    v: Tensor<D>,
    t: usize,
}

impl<D: Dimension> Adam<D> {
    pub fn new(beta_1: f32, beta_2: f32, eta: f32) -> Self {
        Self {
            beta_1,
            beta_2,
            eta,
            m: Tensor::new_filled(0.0),
            v: Tensor::new_filled(0.0),
            t: 1,
        }
    }
}

impl<D: Dimension> Optimizer<D> for Adam<D> {
    fn update(&mut self, w: &mut Tensor<D>, g: Tensor<D>) {
        self.m *= self.beta_1;
        self.m += &(g.clone() * (1.0 - self.beta_1));
        self.v *= self.beta_2;
        self.v += &(g.map_each(|i| i * i) * (1.0 - self.beta_2));

        let m_hat = 1.0 - self.beta_1.powi(self.t as _);
        let v_hat = 1.0 - self.beta_2.powi(self.t as _);

        *w -= &self.m.clone().map_zip_ref(&self.v, |m, v| (m / m_hat) / ((v / v_hat).sqrt() + f32::EPSILON) * self.eta);

        self.t += 1;
    }
}
