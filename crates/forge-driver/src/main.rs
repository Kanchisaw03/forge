mod bench;
mod compile;
#[cfg(feature = "python")]
mod python;

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use crate::bench::{
    print_benchmark_header, run_matmul_bench, theoretical_peak_gflops, BenchConfig,
};
use crate::compile::{check_file, compile_file, dump_stage, DumpStage};

#[derive(Parser, Debug)]
#[command(name = "forge", version, about = "Forge kernel compiler and runtime driver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compile {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    Run {
        input: PathBuf,
        #[arg(long)]
        grid: String,
        #[arg(long)]
        block: String,
        #[arg(long)]
        args: Vec<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Bench {
        input: PathBuf,
        #[arg(long, default_value_t = 256)]
        size: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
    },
    Check {
        input: PathBuf,
    },
    Dump {
        input: PathBuf,
        #[arg(long)]
        stage: String,
    },
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compile { input, output } => {
            let out = compile_file(&input, output.as_deref())?;
            println!("{}", out.display());
        }
        Commands::Run {
            input,
            grid,
            block,
            args,
            output,
        } => {
            let out = compile_file(&input, None)?;
            let (gx, gy) = parse_pair(&grid).context("invalid --grid, expected x,y")?;
            let (bx, by) = parse_pair(&block).context("invalid --block, expected x,y")?;
            println!(
                "compiled {} with grid=({}, {}) block=({}, {}) args={} output={}",
                out.display(),
                gx,
                gy,
                bx,
                by,
                args.len(),
                output
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<none>".to_owned())
            );
        }
        Commands::Bench {
            input: _input,
            size: _size,
            warmup,
            iters,
        } => {
            print_benchmark_header();

            let sizes = [256usize, 512, 1024];
            let mut results = Vec::with_capacity(sizes.len());
            for &n in &sizes {
                let result = run_matmul_bench(BenchConfig { warmup, iters }, n);
                println!(
                    "Size: {n}x{n} | Latency: {:.2} ms | GFLOPS: {:.1}",
                    result.mean_ms, result.gflops
                );
                println!(
                    "FORGE_RESULT: size={n}x{n} latency_ms={:.2} gflops={:.1}",
                    result.mean_ms, result.gflops
                );
                results.push((n, result));
            }

            let theoretical = theoretical_peak_gflops();
            println!("==========================================");
            println!("Size        Forge    Theoretical   % Peak");
            println!("==========================================");
            for (n, result) in results {
                let pct = (result.gflops / theoretical) * 100.0;
                println!(
                    "{:<10} {:>6.1} GF {:>8.1} GF {:>8.1}%",
                    format!("{n}x{n}"),
                    result.gflops,
                    theoretical,
                    pct
                );
            }
            println!("==========================================");
        }
        Commands::Check { input } => {
            check_file(&input)?;
            println!("ok");
        }
        Commands::Dump { input, stage } => {
            let stage = DumpStage::parse(&stage)
                .ok_or_else(|| anyhow!("invalid stage `{stage}`"))?;
            let output = dump_stage(&input, stage)?;
            println!("{output}");
        }
        Commands::Info => {
            print_cpu_info();
            print_benchmark_header();
        }
    }
    Ok(())
}

fn parse_pair(s: &str) -> Result<(u32, u32)> {
    let (a, b) = s
        .split_once(',')
        .ok_or_else(|| anyhow!("missing comma in pair"))?;
    let x = a.parse::<u32>()?;
    let y = b.parse::<u32>()?;
    Ok((x, y))
}

fn print_cpu_info() {
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "avx2={} fma={} sse4.2={}",
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("sse4.2")
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("non-x86_64 target");
    }
}
