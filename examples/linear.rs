use smolmatrix::*;
use smolnn2::*;

fn main() {
    let mut data = Vec::new();
    for i in 0..=100 {
        let x = i as f32 / 10.0 - 5.0;
        data.push((vector!(1 [x]), gen_data(x)));
    }

    let mut l1 = fcnn::Fcnn::new_xavier_uniform(fastrand::f32);
    let mut o1 = fcnn::make_optimizers!(adam::Adam::new(0.9, 0.999, 0.01));
    let mut c1 = fcnn::FcnnCollector::new();

    loop {
        c1.reset();
        let mut err = 0.0;

        for (i, o) in data.iter() {
            let o1 = l1.forward(i);

            let d_err = loss::squared_error_derivative(o1.clone(), o);
            err += loss::squared_error(o1, o)[0];

            l1.back_propagate(&mut c1, d_err, i);
        }

        c1 /= data.len() as f32;
        l1.update(&c1, &mut o1);

        println!("error: {}", err / data.len() as f32);
    }
}

fn gen_data(x: f32) -> Vector<1> {
    vector!(1 [3.0 * x])
}
