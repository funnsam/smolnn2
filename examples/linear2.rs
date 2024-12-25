use smolmatrix::*;
use smolnn2::*;

fn main() {
    let mut l0 = fcnn::Fcnn::<1, 1>::new(|| fastrand::f32() * 0.5);
    let mut l1 = fcnn::Fcnn::<1, 1>::new(|| fastrand::f32() * 0.5);

    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 10.0 - 5.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    loop {
        let mut c0 = fcnn::FcnnCollector::new();
        let mut c1 = fcnn::FcnnCollector::new();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let o0 = l0.forward(i);
            let o1 = l1.forward(&o0);

            let d_err = squared_error_derivative(o1.clone(), o);
            err += squared_error(o1, o)[0];

            let d1 = l1.derivative(&d_err);

            l0.back_propagate(&mut c0, d1, i);
            l1.back_propagate(&mut c1, d_err, &o0);
        }

        l0.collect(c0, 0.01, data.len());
        l1.collect(c1, 0.01, data.len());

        println!("error: {}", err / data.len() as f32);
        // println!("{l0:?}");
        // println!("{l1:?}");
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [3.0 * x])
}
