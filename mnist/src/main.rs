use smolmatrix::*;
use smolnn2::*;

mod reader;

const BAR_LENGTH: usize = 15;
const MINI_BATCH_SIZE: usize = 10000;

model! {
    #[derive(Clone)]
    pub MnistModel: 1, 784 => 1, 10

    => fcnn::Fcnn<784, 64>
    => activation::Tanh
    => fcnn::Fcnn<64, 64>
    => activation::Tanh
    => fcnn::Fcnn<64, 10>
    => activation::Softmax
}

fn main() {
    let (images, labels) = reader::read_data("train", None).unwrap();

    let x1 = (6.0_f32 / (784.0 + 64.0)).sqrt();
    let x3 = (6.0_f32 / (64.0 + 64.0)).sqrt();
    let x5 = (6.0_f32 / (64.0 + 10.0)).sqrt();

    let mut model = MnistModel {
        l1: fcnn::Fcnn::new(|| fastrand::f32() * 2.0 * x1 - x1),
        l2: activation::Tanh,
        l3: fcnn::Fcnn::new(|| fastrand::f32() * 2.0 * x3 - x3),
        l4: activation::Tanh,
        l5: fcnn::Fcnn::new(|| fastrand::f32() * 2.0 * x5 - x5),
        l6: activation::Softmax,
    };

    let mut opt_l1w = adam::Adam::new(0.9, 0.999, 0.001);
    let mut opt_l1b = adam::Adam::new(0.9, 0.999, 0.001);
    let mut opt_l3w = adam::Adam::new(0.9, 0.999, 0.001);
    let mut opt_l3b = adam::Adam::new(0.9, 0.999, 0.001);
    let mut opt_l5w = adam::Adam::new(0.9, 0.999, 0.001);
    let mut opt_l5b = adam::Adam::new(0.9, 0.999, 0.001);

    for i in 1..=100 {
        let mut c1 = fcnn::FcnnCollector::new();
        let mut c3 = fcnn::FcnnCollector::new();
        let mut c5 = fcnn::FcnnCollector::new();
        let mut loss = 0.0;

        for _ in 0..MINI_BATCH_SIZE {
            let i = fastrand::usize(0..images.len());
            let input = &images[i];
            let label = labels[i];
            let mut expected = Vector::new_zeroed();
            expected[label as usize] = 1.0;

            let out = model.back_propagate(input, &expected, loss::categorical_cross_entropy_derivative, &mut c1, &mut (), &mut c3, &mut (), &mut c5, &mut ());
            loss += loss::categorical_cross_entropy(out, &expected).inner.iter().flatten().sum::<f32>();
        }

        opt_l1w.update(&mut model.l1.weight, c1.weight / MINI_BATCH_SIZE as f32);
        opt_l1b.update(&mut model.l1.bias  , c1.bias   / MINI_BATCH_SIZE as f32);
        opt_l3w.update(&mut model.l3.weight, c3.weight / MINI_BATCH_SIZE as f32);
        opt_l3b.update(&mut model.l3.bias  , c3.bias   / MINI_BATCH_SIZE as f32);
        opt_l5w.update(&mut model.l5.weight, c5.weight / MINI_BATCH_SIZE as f32);
        opt_l5b.update(&mut model.l5.bias  , c5.bias   / MINI_BATCH_SIZE as f32);

        println!("{i:>5} {}", loss / MINI_BATCH_SIZE as f32);
    }

    for (i, l) in images.iter().zip(labels.iter()) {
        visualize(i);
        println!("Expected: {}", l);

        let f = model.forward(i);
        let p = f
            .inner
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1[0]
                    .partial_cmp(&b.1[0])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap();

        println!("Predicted: {} ({:.1}%)", p.0, p.1[0] * 100.0);
        bar(&f);
        std::thread::sleep_ms(1000);
    }
}

fn visualize(fb: &Vector<784>) {
    for yhalf in 0..28 / 2 {
        for x in 0..28 {
            plot(fb[(0, x + yhalf * 56)], fb[(0, x + yhalf * 56 + 28)]);
        }

        println!("\x1b[0m");
    }
}

fn plot(a: f32, b: f32) {
    let a = (a * 255.0) as u8;
    let b = (b * 255.0) as u8;
    print!("\x1b[38;2;{a};{a};{a}m\x1b[48;2;{b};{b};{b}m▀");
}

fn bar(v: &Vector<10>) {
    for (i, [v]) in v.inner.iter().enumerate() {
        let len = ((v + 1.0).log2() * BAR_LENGTH as f32).floor() as usize;
        println!(
            "{i} {:━<len$}{:<2$} ({3:.1}%)",
            "",
            "",
            BAR_LENGTH - len,
            v * 100.0
        );
    }
}
