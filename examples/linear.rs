use smolmatrix::*;
use smolnn2::*;

fn main() {
    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 10.0 - 5.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    let mut l0 = fcnn::Fcnn::<1, 1>::new(|| fastrand::f32() * 0.5);
    let mut opt_l0w = adam::Adam::new(0.9, 0.999, 1.0 / data.len() as f32);
    let mut opt_l0b = adam::Adam::new(0.9, 0.999, 1.0 / data.len() as f32);

    loop {
        let mut c0 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let o0 = l0.forward(i);

            let d_err = squared_error_derivative(o0.clone(), o);
            err += squared_error(o0, o)[0];

            l0.back_propagate(&mut c0, d_err, i);
        }

        opt_l0w.update(&mut l0.weight, c0.weight);
        opt_l0b.update(&mut l0.bias, c0.bias);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [3.0 * x])
}
