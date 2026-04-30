mod bench;
mod compile;
#[cfg(feature = "python")]
mod python;

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use crate::bench::{run_matmul_bench, BenchConfig};
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
            warmup,
            iters,
        } => {
            let result = run_matmul_bench(BenchConfig { warmup, iters }, 256);
            println!(
                "latency_ms={:.3} gflops={:.3}",
                result.latency_ms, result.gflops
            );
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
