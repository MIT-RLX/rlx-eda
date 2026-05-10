use spike_dado_r2r::{random_design, run, score_synth, Design, Rng};

fn main() {
    let mut tr = Rng::new(13);
    let target: Design = random_design(&mut tr);
    let score = move |x: &Design| score_synth(x, &target);
    for k in [64, 100, 200] {
        for alpha in [0.5_f64, 0.1, 0.01] {
            for tau in [1.0_f64, 0.5, 2.0] {
                let mut dado_finals = vec![];
                let mut eda_finals = vec![];
                for s in 0..6 {
                    let seed = (s as u32) * 2 + 101;
                    let dado = run(&score, 60, k, tau, alpha, true, seed, &[]);
                    let eda = run(&score, 60, k, tau, alpha, false, seed, &[]);
                    dado_finals.push(*dado.best.last().unwrap());
                    eda_finals.push(*eda.best.last().unwrap());
                }
                let dm: f64 = dado_finals.iter().sum::<f64>() / 6.0;
                let em: f64 = eda_finals.iter().sum::<f64>() / 6.0;
                println!("K={k} alpha={alpha} tau={tau} | DADO {dm:.2} EDA {em:.2} gap {:.2}", dm - em);
            }
        }
    }
}
