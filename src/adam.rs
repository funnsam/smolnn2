use crate::*;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Adam<const W: usize, const H: usize> {
    pub beta_1: f32,
    pub beta_2: f32,
    pub eta: f32,
    m: Matrix<W, H>,
    v: Matrix<W, H>,
    t: usize,
}

impl<const W: usize, const H: usize> Adam<W, H> {
    pub fn new(beta_1: f32, beta_2: f32, eta: f32) -> Self {
        Self {
            beta_1,
            beta_2,
            eta,
            m: Matrix::new_zeroed(),
            v: Matrix::new_zeroed(),
            t: 1,
        }
    }
}

impl<const W: usize, const H: usize> Optimizer<W, H> for Adam<W, H> {
    fn update(&mut self, w: &mut Matrix<W, H>, g: Matrix<W, H>) {
        self.m *= self.beta_1;
        self.m += &(g.clone() * (1.0 - self.beta_1));
        self.v *= self.beta_2;
        self.v += &(g.map_each(|i| *i *= *i) * (1.0 - self.beta_2));

        let m_hat = self.m.clone() / (1.0 - self.beta_1.powi(self.t as _));
        let v_hat = self.v.clone() / (1.0 - self.beta_2.powi(self.t as _));

        *w -= &m_hat.map_zip_ref(&v_hat, |(m, v)| *m = *m / (v.sqrt() + f32::EPSILON) * self.eta);

        self.t += 1;
    }
}
